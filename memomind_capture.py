"""Intake, confirmation, entity extraction."""

import re
import uuid
from datetime import datetime, timezone

import memomind_db as db
import memomind_persona as persona
import memomind_tasks_events as te
from memomind_models import complete, extract_json_from_response, NoModelFoundError

ENTITY_TYPES = ("person", "place", "food", "date", "thing", "topic", "symptom")

CLEAN_SYSTEM = """You clean spoken or typed notes into clear, concise memory entries.
Fix transcription errors, complete fragments, preserve meaning. Return ONLY the cleaned text, nothing else."""

EXTRACT_SYSTEM = """Extract entities from the text. Return JSON array only:
[{"type": "person|place|food|date|thing|topic|symptom", "value": "..."}]
Extract people, places, foods, dates, topics, symptoms. Empty array if none."""

TYPE_SYSTEM = """Classify this memory entry. Return JSON only:
{"type": "note|fact|task|event|thread_item|image", "reason": "brief"}
- fact: durable personal truth
- task: something to do
- event: scheduled occurrence with date/time
- thread_item: part of ongoing project
- note: general observation"""


def clean_input(raw: str) -> str:
    raw = raw.strip()
    if not raw:
        return raw
    try:
        cleaned = complete(
            f"Clean this input for storage as a personal memory:\n\n{raw}",
            system=CLEAN_SYSTEM,
            max_tokens=256,
        )
        return cleaned.strip() or _basic_clean(raw)
    except (NoModelFoundError, Exception):
        return _basic_clean(raw)


def _basic_clean(raw: str) -> str:
    text = raw.strip()
    text = re.sub(r"\s+", " ", text)
    text = text[0].upper() + text[1:] if text else text
    return text


def extract_entities(text: str) -> list[dict]:
    try:
        result = complete(
            f"Extract entities from:\n\n{text}",
            system=EXTRACT_SYSTEM,
            max_tokens=512,
        )
        parsed = extract_json_from_response(result)
        if isinstance(parsed, list):
            entities = []
            for e in parsed:
                if isinstance(e, dict) and e.get("value"):
                    ent_type = e.get("type", "thing")
                    if ent_type not in ENTITY_TYPES:
                        ent_type = "thing"
                    entities.append({
                        "type": ent_type,
                        "value": e["value"],
                        "normalized": e["value"].lower().strip(),
                    })
            return entities
    except (NoModelFoundError, Exception):
        pass
    return _heuristic_entities(text)


def _heuristic_entities(text: str) -> list[dict]:
    entities = []
    food_words = ["shellfish", "peanut", "dairy", "gluten", "nuts", "fish", "egg", "soy", "wheat"]
    lower = text.lower()
    for food in food_words:
        if food in lower:
            entities.append({"type": "food", "value": food, "normalized": food})
    return entities


def detect_type(text: str) -> str:
    kind = te.detect_capture_kind(text)
    if kind in ("task", "event"):
        return kind
    try:
        result = complete(
            f"Classify:\n\n{text}",
            system=TYPE_SYSTEM,
            max_tokens=128,
        )
        parsed = extract_json_from_response(result)
        if isinstance(parsed, dict) and parsed.get("type"):
            t = parsed["type"]
            if t in ("note", "fact", "event", "thread_item", "task", "image"):
                return t
    except (NoModelFoundError, Exception):
        pass
    lower = text.lower()
    if any(w in lower for w in ("can't", "cannot", "allergic", "never", "always")):
        return "fact"
    return "note"


def suggest_thread(text: str, entities: list[dict]) -> dict | None:
    threads = db.list_threads(status="active")
    if not threads:
        return None
    text_lower = text.lower()
    for thread in threads[:20]:
        title_lower = thread["title"].lower()
        if title_lower in text_lower or any(w in text_lower for w in title_lower.split() if len(w) > 3):
            return thread
    return None


def check_conflicts(text: str, entities: list[dict]) -> dict | None:
    similar = db.find_similar_entries(text)
    for entry in similar:
        overlap = _text_overlap(text, entry["content"])
        if overlap > 0.4:
            date = entry["created_at"][:10]
            return {
                "exists": True,
                "entry_id": entry["id"],
                "message": persona.CAPTURE["conflict"].format(date=date),
                "existing_content": entry["content"],
            }
    return None


def _text_overlap(a: str, b: str) -> float:
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / max(len(words_a), len(words_b))


def _build_extra_data(suggested_type: str, cleaned: str) -> dict:
    extra = {"capture_kind": suggested_type}
    if suggested_type == "task":
        extra["task"] = {
            "title": cleaned,
            "priority": "medium",
            "due_date": None,
            "recurrence": None,
            "reminders": [],
        }
    elif suggested_type == "event":
        event_type = te.detect_event_type(cleaned)
        extra["event"] = {
            "title": cleaned,
            "event_date": datetime.now(timezone.utc).replace(hour=14, minute=0, second=0).isoformat(),
            "all_day": False,
            "location": None,
            "notes": None,
            "event_type": event_type,
            "reminders": te.default_reminders_for_event(event_type),
        }
    return extra


def build_confirmation(raw_input: str, input_type: str = "text") -> dict:
    cleaned = clean_input(raw_input)
    entities = extract_entities(cleaned)
    suggested_type = detect_type(cleaned)
    suggested_thread = suggest_thread(cleaned, entities)
    conflict = check_conflicts(cleaned, entities)
    extra_data = _build_extra_data(suggested_type, cleaned)

    conf_id = str(uuid.uuid4())
    payload = {
        "id": conf_id,
        "raw_input": raw_input,
        "cleaned": cleaned,
        "entities": entities,
        "suggested_type": suggested_type,
        "suggested_thread": suggested_thread,
        "conflict": conflict,
        "input_type": input_type,
        "extra_data": extra_data,
    }
    db.save_pending_confirmation(payload)

    return {
        "confirmation_id": conf_id,
        "cleaned": cleaned,
        "entities": [{"type": e["type"], "value": e["value"]} for e in entities],
        "suggested_type": suggested_type,
        "suggested_thread": {
            "id": suggested_thread["id"],
            "title": suggested_thread["title"],
        } if suggested_thread else None,
        "conflict": conflict,
        "extra_data": extra_data,
        "persona_message": persona.CAPTURE["confirm"],
    }


def confirm_capture(
    confirmation_id: str,
    approved_content: str,
    thread_id: str | None = None,
    scope: str = "personal",
    update_entry_id: str | None = None,
    save_as: str | None = None,
    task_data: dict | None = None,
    event_data: dict | None = None,
    reminders: list[dict] | None = None,
) -> dict:
    pending = db.get_pending_confirmation(confirmation_id)
    if not pending:
        raise ValueError("Confirmation not found or expired")

    save_as = save_as or pending.get("suggested_type", "note")
    if save_as in ("task", "event") and not update_entry_id:
        if save_as == "task":
            td = task_data or (pending.get("extra_data") or {}).get("task", {})
            result = te.create_task_from_confirmation(
                title=approved_content,
                notes=td.get("notes"),
                priority=td.get("priority", "medium"),
                due_date=td.get("due_date"),
                recurrence=td.get("recurrence"),
                reminders=reminders or td.get("reminders"),
                thread_id=thread_id,
                raw_input=pending["raw_input"],
            )
            db.delete_pending_confirmation(confirmation_id)
            return {
                "entry_id": result["entry"]["id"],
                "task_id": result["task"]["id"],
                "stored": True,
                "persona_message": persona.TASKS["saved"],
            }
        ed = event_data or (pending.get("extra_data") or {}).get("event", {})
        result = te.create_event_from_confirmation(
            title=approved_content,
            event_date=ed.get("event_date", datetime.now(timezone.utc).isoformat()),
            end_date=ed.get("end_date"),
            location=ed.get("location"),
            all_day=ed.get("all_day", False),
            notes=ed.get("notes"),
            recurrence=ed.get("recurrence"),
            reminders=reminders or ed.get("reminders"),
            thread_id=thread_id,
            raw_input=pending["raw_input"],
        )
        db.delete_pending_confirmation(confirmation_id)
        return {
            "entry_id": result["entry"]["id"],
            "event_id": result["event"]["id"],
            "stored": True,
            "persona_message": persona.EVENTS["saved"],
        }

    entities = pending.get("entities", [])
    entry_type = save_as if save_as in ("note", "fact", "thread_item", "image") else "note"
    conflict = pending.get("conflict")

    if conflict and conflict.get("entry_id") and not update_entry_id:
        update_entry_id = conflict["entry_id"]

    if update_entry_id:
        entry = db.update_entry(
            update_entry_id,
            content=approved_content,
            type=entry_type,
            scope=scope,
            thread_id=thread_id,
            staged=0,
        )
        persona_msg = persona.CAPTURE["updated"]
    else:
        entry = db.create_entry(
            content=approved_content,
            raw_input=pending["raw_input"],
            entry_type=entry_type,
            scope=scope,
            thread_id=thread_id,
            staged=True,
        )
        db.store_entities(entry["id"], entities)
        persona_msg = persona.CAPTURE["saved"]

    db.delete_pending_confirmation(confirmation_id)
    return {
        "entry_id": entry["id"],
        "stored": True,
        "persona_message": persona_msg,
        "updated": bool(update_entry_id),
    }


def discard_capture(confirmation_id: str) -> dict:
    pending = db.get_pending_confirmation(confirmation_id)
    if pending:
        db.delete_pending_confirmation(confirmation_id)
    return {"discarded": True, "persona_message": persona.CAPTURE["discarded"]}
