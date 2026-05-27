"""Consolidation, decay, strengthening, hippocampus."""

import json
import logging
from datetime import datetime, timezone, timedelta

import memomind_db as db
from memomind_models import complete, extract_json_from_response, NoModelFoundError

logger = logging.getLogger("memomind.brain")

BASE_DECAY = 0.02
ACCESS_BOOST = 1.5
MIN_STRENGTH = 0.1
PIN_STRENGTH = 1.0
DUPLICATE_THRESHOLD = 0.85


def decay_entry(entry: dict, days_since_access: float) -> float:
    if entry.get("pinned"):
        return PIN_STRENGTH
    decayed = entry["strength"] * (1 - BASE_DECAY) ** days_since_access
    return max(decayed, MIN_STRENGTH)


def boost_on_access(entry: dict) -> float:
    boosted = min(entry["strength"] * ACCESS_BOOST, 1.0)
    return boosted


def _days_since(iso_date: str | None) -> float:
    if not iso_date:
        return 999
    try:
        dt = datetime.fromisoformat(iso_date.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds() / 86400
    except (ValueError, TypeError):
        return 999


def _text_similarity(a: str, b: str) -> float:
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / max(len(words_a), len(words_b))


def find_duplicate(entry: dict) -> dict | None:
    similar = db.find_similar_entries(entry["content"], limit=10)
    for other in similar:
        if other["id"] == entry["id"]:
            continue
        if _text_similarity(entry["content"], other["content"]) > DUPLICATE_THRESHOLD:
            return other
    return None


def merge_into(source: dict, target: dict):
    combined = f"{target['content']}\n{source['content']}"
    db.update_entry(target["id"], content=combined, strength=1.0, staged=0)
    db.soft_delete_entry(source["id"])


def promote_to_longterm(entry: dict):
    db.update_entry(entry["id"], staged=0)


def stage_entry(entry_id: str):
    db.update_entry(entry_id, staged=1)


def process_staged():
    staged = db.get_staged_entries()
    for entry in staged:
        duplicate = find_duplicate(entry)
        if duplicate:
            merge_into(entry, duplicate)
        else:
            promote_to_longterm(entry)


def decay_all():
    entries = db.get_all_active_entries()
    for entry in entries:
        if entry.get("pinned"):
            continue
        days = _days_since(entry.get("last_accessed") or entry.get("created_at"))
        new_strength = decay_entry(entry, days)
        if abs(new_strength - entry["strength"]) > 0.001:
            db.update_entry(entry["id"], strength=new_strength)


def reinforce_connections():
    connections = db.list_all_connections()
    for conn in connections:
        if conn.get("last_reinforced"):
            days = _days_since(conn["last_reinforced"])
            if days < 7:
                db.reinforce_connection(conn["entry_a"], conn["entry_b"], 0.05)


def merge_duplicates():
    entries = db.get_all_active_entries()
    seen = set()
    for entry in entries:
        if entry["id"] in seen:
            continue
        dup = find_duplicate(entry)
        if dup and dup["id"] not in seen:
            merge_into(entry, dup)
            seen.add(entry["id"])
            seen.add(dup["id"])


def refresh_thread_summaries():
    for thread in db.list_threads(status="active"):
        entries = db.get_thread_entries(thread["id"])
        if not entries:
            continue
        content = "\n".join(e["content"] for e in entries[-10:])
        try:
            summary = complete(
                f"Summarize this thread in 1-2 sentences:\n\n{content}",
                system="Return only the summary text.",
                max_tokens=128,
            )
            db.update_thread(thread["id"], summary=summary.strip())
        except (NoModelFoundError, Exception):
            db.update_thread(thread["id"], summary=content[:200])


def refresh_fact_summaries():
    entries = db.get_all_active_entries()
    facts_by_topic: dict[str, list] = {}
    for entry in entries:
        if entry.get("type") == "fact":
            ents = db.get_entities_for_entry(entry["id"])
            topic = ents[0]["value"] if ents else "general"
            facts_by_topic.setdefault(topic, []).append(entry)

    for topic, topic_entries in facts_by_topic.items():
        content = "\n".join(e["content"] for e in topic_entries)
        try:
            summary = complete(
                f"Synthesize facts about '{topic}':\n\n{content}",
                system="Return one concise factual summary.",
                max_tokens=256,
            )
            db.upsert_fact(
                topic,
                summary.strip(),
                [e["id"] for e in topic_entries],
            )
        except (NoModelFoundError, Exception):
            db.upsert_fact(topic, content[:300], [e["id"] for e in topic_entries])


def flag_stale(older_than_days: int = 180):
    stale = db.get_stale_entries(older_than_days)
    for entry in stale:
        db.set_meta(f"stale_{entry['id']}", "1")


def rerank_fts5():
    try:
        with db._connect() as conn:
            conn.execute("INSERT INTO entries_fts(entries_fts) VALUES('rebuild')")
    except Exception as e:
        logger.warning("FTS rebuild failed: %s", e)


def consolidate():
    logger.info("Starting consolidation...")
    process_staged()
    merge_duplicates()
    reinforce_connections()
    db.prune_connections(threshold=0.1)
    decay_all()
    refresh_thread_summaries()
    refresh_fact_summaries()
    flag_stale(older_than_days=180)
    rerank_fts5()
    db.set_meta("last_consolidation", db.now_iso())
    logger.info("Consolidation complete")


def get_related_entries(entry_id: str, depth: int = 2) -> list[dict]:
    entry = db.get_entry(entry_id)
    if not entry:
        return []
    activated = __import__("memomind_search", fromlist=["spreading_activation"]).spreading_activation(
        [entry], depth=depth
    )
    results = []
    for eid, score in activated:
        if eid == entry_id:
            continue
        e = db.get_entry(eid)
        if e:
            e["activation_score"] = score
            results.append(e)
    return results
