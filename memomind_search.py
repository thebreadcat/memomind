"""Map-reduce search, tools, spreading activation."""

import json
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
    stop = {"what", "cant", "can't", "can", "i", "my", "the", "a", "an", "is", "are", "do", "does", "how", "when", "where", "who", "why", "about", "know", "tell", "me"}
    words = []
    for w in query.lower().replace("'", "").split():
        w = w.strip("?.,!")
        if len(w) > 2 and w not in stop:
            words.append(w)
    return words or [query.strip("?.,!")]


def prefilter(query: str, scope: str = "personal", limit: int = 100) -> list[dict]:
    fts_results = []
    for term in _query_terms(query):
        fts_results.extend(db.fts_search(term, limit=50))
    if not fts_results:
        fts_results = db.fts_search(query, limit=200)

    entity_norms = extract_query_entities(query)
    entity_results = db.search_by_entities(entity_norms, limit=50)

    # Include all entries when DB is small
    if db.count_entries(scope=scope) <= 20:
        all_entries = db.list_entries(limit=20, scope=scope)
        fts_results = fts_results + all_entries

    seen = set()
    combined = []
    for entry in fts_results + entity_results:
        if entry["id"] in seen:
            continue
        if scope and entry.get("scope") != scope:
            continue
        seen.add(entry["id"])
        combined.append(entry)

    combined.sort(
        key=lambda e: (e.get("strength", 1.0), e.get("created_at", "")),
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


def _unique_finding_texts(all_findings: list[dict]) -> list[str]:
    """Distinct finding strings, preserving order."""
    texts: list[str] = []
    seen: set[str] = set()
    for f in all_findings:
        t = (f.get("finding") or "").strip()
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
            all_findings.append({**finding, "chunk": i + 1})
            for eid in finding.get("entry_ids", []):
                processed_ids.add(eid)
                db.boost_entry_access(eid)
        all_contradictions.extend(result.get("contradictions", []))

    # Spreading activation on top findings
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

    return {
        "answer": reduced["answer"],
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
            all_findings.append({**finding, "chunk": i + 1})
            for eid in finding.get("entry_ids", []):
                processed_ids.add(eid)
                db.boost_entry_access(eid)
        all_contradictions.extend(result.get("contradictions", []))
        yield {"event": "chunk", "data": result}

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

    yield {
        "event": "complete",
        "data": {
            "answer": reduced["answer"],
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
