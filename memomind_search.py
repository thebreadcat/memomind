"""Map-reduce search, tools, spreading activation."""

import json
import re
import difflib
from datetime import datetime, timezone

import memomind_db as db
import memomind_persona as persona
from memomind_config import load_config
from memomind_models import complete, extract_json_from_response, get_provider, NoModelFoundError

CHUNK_SYNTHESIS_SYSTEM = """You analyze personal memory entries against a user query.
Return JSON only:
{
  "findings": [{"finding": "...", "entry_ids": ["id1"], "confidence": "high|medium|low"}],
  "uncertain": [],
  "contradictions": [{"description": "...", "entry_ids": ["id1", "id2"]}]
}
Be honest. Only state what entries support. Flag contradictions.
If multiple entries each answer the query, return a separate finding for EACH entry — do not merge distinct facts into one finding."""

REDUCE_SYSTEM = """Synthesize findings into a warm, honest answer for the user.
Start with facts from their records. Never invent.
You MUST include every distinct finding below — if there are several, mention each one (short list or combined sentences). Do not drop any finding.
Return JSON:
{
  "answer": "full answer text",
  "confidence": "high|medium|low|none",
  "contradictions": []
}
If nothing relevant, answer should say you don't have anything on that."""


def extract_query_entities(query: str) -> list[str]:
    from memomind_capture import extract_entities
    ents = extract_entities(query)
    return [e["normalized"] for e in ents]


def _query_terms(query: str) -> list[str]:
    stop = {
        "what", "cant", "can't", "can", "i", "my", "the", "a", "an", "is", "are",
        "do", "does", "how", "when", "where", "who", "why", "about", "know", "tell", "me",
        "any", "some", "should", "would", "could",
    }
    words = []
    for w in query.lower().replace("'", "").split():
        w = w.strip("?.,!")
        if len(w) > 2 and w not in stop:
            words.append(w)
    raw = query.strip("?.,! ").lower()
    if not words and raw:
        return [raw]
    return words


def _entry_tokens(entry: dict) -> set[str]:
    """Tokenize entry content into lowercase words for fuzzy matching."""
    content = (entry.get("content") or "").lower()
    return {t for t in re.findall(r"[a-z0-9]+", content) if len(t) >= 3}


def _candidate_vocab(scope: str = "personal", limit: int = 300) -> set[str]:
    """
    Build a lightweight vocabulary from recent entries.
    Kept bounded for performance on local devices.
    """
    vocab: set[str] = set()
    for e in db.list_entries(limit=limit, scope=scope):
        vocab.update(_entry_tokens(e))
    return vocab


def _expand_terms_with_typos(terms: list[str], scope: str = "personal") -> list[str]:
    """
    Expand query terms with close spellings found in memory content.
    Example: saop -> soap
    """
    if not terms:
        return terms
    vocab = _candidate_vocab(scope=scope)
    if not vocab:
        return terms

    expanded = list(terms)
    for t in terms:
        if t in vocab:
            continue
        # Allow common transposition typos (saop -> soap) while staying conservative.
        matches = difflib.get_close_matches(t, list(vocab), n=2, cutoff=0.74)
        for m in matches:
            if m not in expanded:
                expanded.append(m)
    return expanded


def _is_broad_query(query: str) -> bool:
    """Open questions that should search widely (e.g. what can't I eat?)."""
    q = query.lower()
    broad_phrases = (
        "what can", "what can't", "what cant", "what do i", "what did i",
        "tell me about", "everything about", "all my", "what do you know",
        "what have i", "remind me",
    )
    return any(p in q for p in broad_phrases)


def _is_specific_query(query: str, terms: list[str]) -> bool:
    if _is_broad_query(query):
        return False
    return 1 <= len(terms) <= 4


def _entry_matches_terms(entry: dict, terms: list[str]) -> bool:
    content = (entry.get("content") or "").lower()
    return any(t in content for t in terms)


def _finding_matches_terms(finding: dict, terms: list[str]) -> bool:
    """Specific query guard: keep only findings tied to the searched terms."""
    if not terms:
        return True
    text = (finding.get("finding") or "").lower()
    if any(t in text for t in terms):
        return True
    for eid in finding.get("entry_ids", []):
        entry = db.get_entry(eid)
        if entry and _entry_matches_terms(entry, terms):
            return True
    return False


def _relevance_score(entry: dict, terms: list[str]) -> float:
    content = (entry.get("content") or "").lower()
    score = float(entry.get("strength", 1.0) or 1.0)
    for t in terms:
        if t not in content:
            continue
        score += 10.0
        if re.search(rf"\b{re.escape(t)}\b", content):
            score += 8.0
    return score


def prefilter(query: str, scope: str = "personal", limit: int = 100) -> list[dict]:
    terms = _expand_terms_with_typos(_query_terms(query), scope=scope)
    specific = _is_specific_query(query, terms)

    fts_results = []
    for term in terms:
        fts_results.extend(db.fts_search(term, limit=50))
    if not fts_results:
        fts_results = db.fts_search(query.strip("?.,! "), limit=200)

    entity_norms = extract_query_entities(query)
    entity_results = db.search_by_entities(entity_norms, limit=50) if not specific else []

    seen = set()
    combined = []
    for entry in fts_results + entity_results:
        if entry["id"] in seen:
            continue
        if scope and entry.get("scope") != scope:
            continue
        if specific and terms and not _entry_matches_terms(entry, terms):
            continue
        seen.add(entry["id"])
        combined.append(entry)

    if specific and terms:
        combined = [e for e in combined if _entry_matches_terms(e, terms)]

    combined.sort(
        key=lambda e: _relevance_score(e, terms),
        reverse=True,
    )
    return combined[:limit]


def split_chunks(entries: list[dict], size: int | None = None) -> list[list[dict]]:
    config = load_config()
    chunk_size = size or config.get("chunk_size", 20)
    return [entries[i : i + chunk_size] for i in range(0, len(entries), chunk_size)]


def _ensure_per_entry_findings(chunk: list[dict], findings: list[dict]) -> list[dict]:
    """If the model merged entries, add one finding per uncovered entry."""
    covered: set[str] = set()
    out = list(findings)
    for f in out:
        for eid in f.get("entry_ids", []):
            covered.add(eid)
    for e in chunk:
        if e["id"] in covered:
            continue
        out.append({
            "finding": e["content"][:500],
            "entry_ids": [e["id"]],
            "confidence": "medium",
        })
        covered.add(e["id"])
    return out


def map_chunk(query: str, chunk: list[dict], chunk_num: int) -> dict:
    entries_text = "\n".join(
        f"[{e['id']}] ({e.get('type', 'note')}) {e['content']}" for e in chunk
    )
    prompt = f"Query: {query}\n\nEntries:\n{entries_text}"
    try:
        result = complete(prompt, system=CHUNK_SYNTHESIS_SYSTEM, max_tokens=1024)
        parsed = extract_json_from_response(result)
        if isinstance(parsed, dict):
            findings = _ensure_per_entry_findings(
                chunk, parsed.get("findings", [])
            )
            return {
                "chunk": chunk_num,
                "findings": findings,
                "uncertain": parsed.get("uncertain", []),
                "contradictions": parsed.get("contradictions", []),
            }
    except (NoModelFoundError, Exception):
        pass
    # Fallback without model — one finding per entry
    return {
        "chunk": chunk_num,
        "findings": [
            {
                "finding": e["content"][:500],
                "entry_ids": [e["id"]],
                "confidence": "medium",
            }
            for e in chunk
        ],
        "uncertain": [],
        "contradictions": [],
    }


def spreading_activation(seed_entries: list[dict], depth: int = 2) -> list[tuple[str, float]]:
    activated: dict[str, float] = {e["id"]: 1.0 for e in seed_entries}

    for d in range(depth):
        decay = 0.5 ** (d + 1)
        current_ids = list(activated.keys())
        for entry_id in current_ids:
            for neighbor_id, conn_strength in db.get_neighbor_entry_ids(entry_id):
                if neighbor_id not in activated:
                    activated[neighbor_id] = conn_strength * decay
                else:
                    activated[neighbor_id] = max(
                        activated[neighbor_id], conn_strength * decay
                    )

    return sorted(activated.items(), key=lambda x: x[1], reverse=True)


def _normalize_sentence(text: str) -> str:
    t = (text or "").strip()
    while ".." in t:
        t = t.replace("..", ".")
    if t and t[-1] not in ".!?":
        t += "."
    return t


def _is_food_query(query: str) -> bool:
    q = query.lower()
    return any(w in q for w in ("eat", "food", "diet", "allerg", "meal", "drink", "consume"))


def _classify_finding(text: str, query: str) -> str:
    """Group finding for at-a-glance display: avoid | ok | notes."""
    t = text.lower()
    if not _is_food_query(query):
        return "notes"

    avoid = (
        "can't", "cannot", "can not", "don't eat", "do not eat", "shouldn't",
        "should not", "not eat", "avoid", "allerg", "hurt", "sick", "poison",
        "no longer", "must not", "we don't eat", "don't eat",
    )
    ok = (
        "fine", "ok to", "okay to", "can eat", "allowed", "doesn't hurt",
        "does not hurt", "safe to", "good to eat", "no problem", "is fine",
    )

    has_avoid = any(s in t for s in avoid)
    has_ok = any(s in t for s in ok)

    if has_avoid and not has_ok:
        return "avoid"
    if has_ok and not has_avoid:
        return "ok"
    if has_avoid:
        return "avoid"
    if has_ok:
        return "ok"
    return "notes"


def organize_findings(all_findings: list[dict], query: str) -> list[dict]:
    """Structured groups for UI — avoids one long paragraph."""
    buckets: dict[str, list[dict]] = {"avoid": [], "ok": [], "notes": []}
    seen: set[str] = set()

    for f in all_findings:
        text = _normalize_sentence(f.get("finding") or "")
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        cat = _classify_finding(text, query)
        buckets[cat].append({
            "text": text,
            "entry_ids": f.get("entry_ids", []),
            "confidence": f.get("confidence", "medium"),
        })

    terms = _query_terms(query)
    if _is_food_query(query) and not _is_specific_query(query, terms):
        labels = {
            "avoid": "Avoid",
            "ok": "OK for you",
            "notes": "Other notes",
        }
        order = ("avoid", "ok", "notes")
    elif _is_specific_query(query, terms) and len(terms) == 1:
        # For typos (e.g. "saop"), prefer a close token from returned findings.
        raw_topic = terms[0]
        token_pool: set[str] = set()
        for group_items in buckets.values():
            for it in group_items:
                token_pool.update(re.findall(r"[a-z0-9]+", it["text"].lower()))
        token_pool = {t for t in token_pool if len(t) >= 3}
        if raw_topic not in token_pool and token_pool:
            close = difflib.get_close_matches(raw_topic, list(token_pool), n=1, cutoff=0.74)
            topic = (close[0] if close else raw_topic).capitalize()
        else:
            topic = raw_topic.capitalize()
        labels = {"notes": f"About {topic}"}
        order = ("notes",)
    else:
        labels = {"notes": "What I found"}
        order = ("notes",)

    groups = []
    for gid in order:
        items = buckets.get(gid, [])
        if items:
            groups.append({"id": gid, "label": labels[gid], "items": items})
    return groups


def search_summary_line(groups: list[dict]) -> str:
    if not groups:
        return ""
    parts = [f"{len(g['items'])} {g['label'].lower()}" for g in groups]
    return " · ".join(parts)


def _focused_answer_from_items(items: list[dict]) -> str:
    """Short merged line for a few on-topic hits (e.g. soap? → buy + don't eat)."""
    neutral, warnings = [], []
    for it in items:
        t = it["text"].lower()
        if any(x in t for x in ("can't", "cannot", "don't eat", "do not eat", "hurt", "sick", "must not")):
            warnings.append(it)
        else:
            neutral.append(it)

    ordered = [_normalize_sentence(x["text"]) for x in neutral + warnings]
    if not ordered:
        return persona.SEARCH["not_found"]
    if len(ordered) == 1:
        body = ordered[0]
    elif len(ordered) == 2:
        a, b = ordered[0].rstrip("."), ordered[1]
        b = b[0].lower() + b[1:] if b else b
        body = f"{a}, and remember {b}"
    else:
        body = ". ".join(ordered[:-1]) + ". " + ordered[-1]

    if body and body[-1] not in ".!?":
        body += "."
    return persona.SEARCH["from_records"] + body[0].lower() + body[1:]


def search_intro(groups: list[dict], query: str, terms: list[str] | None = None) -> str:
    if not groups:
        return persona.SEARCH["not_found"]

    terms = terms or _query_terms(query)
    all_items = []
    for g in groups:
        all_items.extend(g["items"])

    if _is_specific_query(query, terms) and 1 <= len(all_items) <= 5:
        return _focused_answer_from_items(all_items)

    if _is_food_query(query):
        avoid_n = sum(len(g["items"]) for g in groups if g["id"] == "avoid")
        ok_n = sum(len(g["items"]) for g in groups if g["id"] == "ok")
        if avoid_n and ok_n:
            return f"From your records — {avoid_n} to avoid, {ok_n} that are fine."
        if avoid_n:
            return f"From your records — {avoid_n} thing{'s' if avoid_n != 1 else ''} to avoid."
        if ok_n:
            return f"From your records — {ok_n} thing{'s' if ok_n != 1 else ''} that are fine."
    total = sum(len(g["items"]) for g in groups)
    return f"From your records — {total} match{'es' if total != 1 else ''}."


def _unique_finding_texts(all_findings: list[dict]) -> list[str]:
    """Distinct finding strings, preserving order."""
    texts: list[str] = []
    seen: set[str] = set()
    for f in all_findings:
        t = _normalize_sentence(f.get("finding") or "")
        if not t:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        texts.append(t)
    return texts


def compose_answer_from_findings(all_findings: list[dict]) -> dict:
    """Deterministic answer that always includes every distinct finding."""
    texts = _unique_finding_texts(all_findings)
    if not texts:
        return {
            "answer": persona.SEARCH["not_found"],
            "confidence": "none",
            "contradictions": [],
        }

    confidences = [f.get("confidence", "medium") for f in all_findings if f.get("finding")]
    if any(c == "high" for c in confidences):
        confidence = "high"
    elif any(c == "medium" for c in confidences):
        confidence = "medium"
    else:
        confidence = confidences[0] if confidences else "medium"

    body = ". ".join(texts)
    if body and body[-1] not in ".!?":
        body += "."

    prefix = persona.SEARCH["from_records"]
    if confidence in ("low", "medium") and len(texts) == 1:
        prefix = persona.SEARCH["uncertain"]
    answer = prefix + body[0].lower() + body[1:] if body else prefix + persona.SEARCH["not_found"]

    return {
        "answer": answer,
        "confidence": confidence,
        "contradictions": [],
    }


def reduce_findings(query: str, all_findings: list[dict], entry_count: int) -> dict:
    if not all_findings:
        return {
            "answer": persona.SEARCH["not_found"],
            "confidence": "none",
            "contradictions": [],
        }

    unique = _unique_finding_texts(all_findings)
    # Multiple distinct facts: compose in Python so nothing is dropped
    if len(unique) > 1:
        return compose_answer_from_findings(all_findings)

    findings_text = json.dumps(all_findings, indent=2)
    prompt = f"Query: {query}\nTotal entries searched: {entry_count}\n\nFindings:\n{findings_text}"

    try:
        result = complete(prompt, system=REDUCE_SYSTEM, max_tokens=1024)
        parsed = extract_json_from_response(result)
        if isinstance(parsed, dict) and parsed.get("answer"):
            answer = parsed["answer"]
            if not answer.lower().startswith(("from your", "based on", "i don't")):
                prefix = persona.SEARCH["from_records"]
                if parsed.get("confidence") in ("low", "medium"):
                    prefix = persona.SEARCH["uncertain"]
                answer = prefix + answer[0].lower() + answer[1:] if answer else answer
            return {
                "answer": answer,
                "confidence": parsed.get("confidence", "medium"),
                "contradictions": parsed.get("contradictions", []),
            }
    except (NoModelFoundError, Exception):
        pass

    return compose_answer_from_findings(all_findings)


def search(query: str, scope: str = "personal", limit: int = 100) -> dict:
    if not query.strip():
        return {
            "answer": "What would you like to know?",
            "confidence": "none",
            "sourced_from": None,
            "chunks_processed": 0,
            "total_entries_searched": 0,
            "supporting": [],
            "also_found": [],
            "contradictions": [],
            "persona_message": None,
        }

    terms = _expand_terms_with_typos(_query_terms(query), scope=scope)
    specific = _is_specific_query(query, terms)
    candidates = prefilter(query, scope=scope, limit=limit)
    total = len(candidates)

    if not total:
        return {
            "answer": persona.SEARCH["not_found"],
            "confidence": "none",
            "sourced_from": "your records",
            "chunks_processed": 0,
            "total_entries_searched": 0,
            "supporting": [],
            "also_found": [],
            "contradictions": [],
            "persona_message": persona.SEARCH["thin"] if db.count_entries() < 3 else None,
        }

    config = load_config()
    max_chunks = config.get("max_chunks", 10)
    chunks = split_chunks(candidates)[:max_chunks]

    all_findings = []
    all_contradictions = []
    processed_ids = set()

    for i, chunk in enumerate(chunks):
        result = map_chunk(query, chunk, i + 1)
        for finding in result.get("findings", []):
            if specific and not _finding_matches_terms(finding, terms):
                continue
            all_findings.append({**finding, "chunk": i + 1})
            for eid in finding.get("entry_ids", []):
                processed_ids.add(eid)
                db.boost_entry_access(eid)
        all_contradictions.extend(result.get("contradictions", []))

    # Spreading activation on top findings
    if not specific:
        seed = candidates[:5]
        activated = spreading_activation(seed, depth=2)
        new_ids = [eid for eid, _ in activated if eid not in processed_ids][:10]
        if new_ids:
            extra_entries = [db.get_entry(eid) for eid in new_ids]
            extra_entries = [e for e in extra_entries if e]
            if extra_entries:
                extra_result = map_chunk(query, extra_entries, len(chunks) + 1)
                for finding in extra_result.get("findings", []):
                    all_findings.append({**finding, "chunk": len(chunks) + 1})
                all_contradictions.extend(extra_result.get("contradictions", []))

    # Reinforce co-accessed connections
    entry_ids = list(processed_ids)[:10]
    for i, a in enumerate(entry_ids):
        for b in entry_ids[i + 1 : i + 3]:
            db.reinforce_connection(a, b)

    reduced = reduce_findings(query, all_findings, total)

    unincluded = [e for e in candidates if e["id"] not in processed_ids]
    also_found = []
    if unincluded:
        also_found.append({
            "summary": f"{len(unincluded)} other related entries",
            "entry_ids": [e["id"] for e in unincluded[:10]],
            "expandable": True,
        })

    supporting = [
        {
            "finding": f.get("finding", ""),
            "entries": f.get("entry_ids", []),
            "chunk": f.get("chunk", 0),
        }
        for f in all_findings
    ]

    enriched_supporting = []
    for s in supporting:
        item = dict(s)
        for eid in s.get("entries", []):
            entry = db.get_entry(eid)
            if entry:
                item["entry_hints"] = item.get("entry_hints", [])
                item["entry_hints"].append(db.enrich_entry(entry))
        enriched_supporting.append(item)

    groups = organize_findings(all_findings, query)
    intro = search_intro(groups, query, terms)
    summary = search_summary_line(groups)

    return {
        "answer": reduced["answer"],
        "answer_short": intro,
        "summary_line": summary,
        "groups": groups,
        "confidence": reduced["confidence"],
        "sourced_from": "your records",
        "chunks_processed": len(chunks),
        "total_entries_searched": total,
        "supporting": enriched_supporting,
        "also_found": also_found,
        "contradictions": all_contradictions + reduced.get("contradictions", []),
        "persona_message": None,
    }


def search_stream(query: str, scope: str = "personal", limit: int = 100):
    """Generator yielding SSE events during search."""
    yield {"event": "start", "data": {"query": query}}

    if not query.strip():
        yield {
            "event": "complete",
            "data": {
                "answer": "What would you like to know?",
                "confidence": "none",
                "total_entries_searched": 0,
            },
        }
        return

    terms = _expand_terms_with_typos(_query_terms(query), scope=scope)
    specific = _is_specific_query(query, terms)
    candidates = prefilter(query, scope=scope, limit=limit)
    yield {"event": "candidates", "data": {"count": len(candidates)}}

    if not candidates:
        yield {
            "event": "complete",
            "data": {
                "answer": persona.SEARCH["not_found"],
                "confidence": "none",
                "sourced_from": "your records",
                "chunks_processed": 0,
                "total_entries_searched": 0,
                "supporting": [],
                "also_found": [],
                "contradictions": [],
            },
        }
        return

    config = load_config()
    chunks = split_chunks(candidates)[: config.get("max_chunks", 10)]
    all_findings = []
    all_contradictions = []
    processed_ids = set()

    for i, chunk in enumerate(chunks):
        result = map_chunk(query, chunk, i + 1)
        for finding in result.get("findings", []):
            if specific and not _finding_matches_terms(finding, terms):
                continue
            all_findings.append({**finding, "chunk": i + 1})
            for eid in finding.get("entry_ids", []):
                processed_ids.add(eid)
                db.boost_entry_access(eid)
        all_contradictions.extend(result.get("contradictions", []))
        yield {"event": "chunk", "data": result}

    if not specific:
        seed = candidates[:5]
        activated = spreading_activation(seed, depth=2)
        new_ids = [eid for eid, _ in activated if eid not in processed_ids][:10]
        if new_ids:
            extra = [db.get_entry(eid) for eid in new_ids]
            extra = [e for e in extra if e]
            if extra:
                extra_result = map_chunk(query, extra, len(chunks) + 1)
                all_findings.extend(
                    {**f, "chunk": len(chunks) + 1} for f in extra_result.get("findings", [])
                )
                all_contradictions.extend(extra_result.get("contradictions", []))

    reduced = reduce_findings(query, all_findings, len(candidates))
    unincluded = [e for e in candidates if e["id"] not in processed_ids]
    also_found = []
    if unincluded:
        also_found.append({
            "summary": f"{len(unincluded)} other related entries",
            "entry_ids": [e["id"] for e in unincluded[:10]],
            "expandable": True,
        })

    groups = organize_findings(all_findings, query)
    yield {
        "event": "complete",
        "data": {
            "answer": reduced["answer"],
            "answer_short": search_intro(groups, query, terms),
            "summary_line": search_summary_line(groups),
            "groups": groups,
            "confidence": reduced["confidence"],
            "sourced_from": "your records",
            "chunks_processed": len(chunks),
            "total_entries_searched": len(candidates),
            "supporting": [
                {
                    "finding": f.get("finding", ""),
                    "entries": f.get("entry_ids", []),
                    "chunk": f.get("chunk", 0),
                }
                for f in all_findings
            ],
            "also_found": also_found,
            "contradictions": all_contradictions,
            "persona_message": None,
        },
    }
