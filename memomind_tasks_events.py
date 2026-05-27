"""Tasks, events, reminders business logic."""

import json
import re
import uuid
from datetime import datetime, timedelta, timezone

import memomind_db as db

EVENT_REMINDER_DEFAULTS = {
    "birthday": [10080, 1440, 480],
    "anniversary": [20160, 1440],
    "appointment": [1440, 120],
    "meeting": [60, 15],
    "bill": [4320, 1440],
    "holiday": [10080, 1440],
    "default": [1440],
}

EVENT_KEYWORDS = {
    "birthday": ["birthday", "bday"],
    "anniversary": ["anniversary"],
    "appointment": ["appointment", "dentist", "doctor"],
    "meeting": ["meeting", "call with"],
    "bill": ["bill", "payment due", "pay "],
    "holiday": ["holiday", "vacation"],
}

TASK_KEYWORDS = ["todo", "task", "remind me to", "need to", "don't forget", "call ", "pay ", "pick up"]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def detect_event_type(title: str, notes: str = "") -> str:
    text = f"{title} {notes}".lower()
    for etype, keywords in EVENT_KEYWORDS.items():
        if any(k in text for k in keywords):
            return etype
    return "default"


def default_reminders_for_event(event_type: str) -> list[dict]:
    offsets = EVENT_REMINDER_DEFAULTS.get(event_type, EVENT_REMINDER_DEFAULTS["default"])
    return [{"offset_minutes": o, "label": None, "method": "notification"} for o in offsets]


def detect_capture_kind(text: str) -> str:
    lower = text.lower()
    if any(k in lower for k in TASK_KEYWORDS) or re.search(r"\b(due|by)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|today|\d)", lower):
        return "task"
    if any(k in lower for k in ["birthday", "anniversary", "appointment", "meeting at", "on ", "schedule"]):
        return "event"
    return "note"


def compute_fire_at(due_or_event_iso: str, offset_minutes: int) -> str:
    dt = datetime.fromisoformat(due_or_event_iso.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    fire = dt - timedelta(minutes=offset_minutes)
    return fire.isoformat()


def link_task_to_memory(entry_id: str, title: str):
    try:
        import memomind_search as search
        related = search.prefilter(title, limit=5)
        for entry in related:
            if entry["id"] != entry_id:
                db.reinforce_connection(entry_id, entry["id"], 0.05)
                db.get_or_create_connection(entry_id, entry["id"])
    except Exception:
        pass


def create_task_from_confirmation(
    title: str,
    notes: str | None = None,
    priority: str = "medium",
    due_date: str | None = None,
    recurrence: dict | None = None,
    reminders: list[dict] | None = None,
    thread_id: str | None = None,
    raw_input: str | None = None,
) -> dict:
    entry = db.create_entry(
        content=title if not notes else f"{title}\n{notes}",
        raw_input=raw_input or title,
        entry_type="task",
        thread_id=thread_id,
        staged=False,
    )
    task = db.create_task(
        entry_id=entry["id"],
        title=title,
        notes=notes,
        priority=priority,
        due_date=due_date,
        recurrence=recurrence,
        thread_id=thread_id,
    )
    due_iso = f"{due_date}T09:00:00+00:00" if due_date and "T" not in due_date else (due_date or now_iso())
    for r in reminders or []:
        db.create_reminder(
            parent_id=task["id"],
            parent_type="task",
            offset_minutes=int(r.get("offset_minutes", 1440)),
            label=r.get("label"),
            method=r.get("method", "notification"),
            fire_at=compute_fire_at(due_iso, int(r.get("offset_minutes", 1440))),
        )
    link_task_to_memory(entry["id"], title)
    return {"entry": entry, "task": task}


def create_event_from_confirmation(
    title: str,
    event_date: str,
    end_date: str | None = None,
    location: str | None = None,
    all_day: bool = False,
    notes: str | None = None,
    recurrence: dict | None = None,
    reminders: list[dict] | None = None,
    thread_id: str | None = None,
    raw_input: str | None = None,
) -> dict:
    content = title
    if location:
        content += f" @ {location}"
    if notes:
        content += f"\n{notes}"
    entry = db.create_entry(
        content=content,
        raw_input=raw_input or title,
        entry_type="event",
        thread_id=thread_id,
        staged=False,
    )
    event = db.create_event(
        entry_id=entry["id"],
        title=title,
        notes=notes,
        event_date=event_date,
        end_date=end_date,
        location=location,
        all_day=all_day,
        recurrence=recurrence,
    )
    for r in reminders or []:
        db.create_reminder(
            parent_id=event["id"],
            parent_type="event",
            offset_minutes=int(r.get("offset_minutes", 1440)),
            label=r.get("label"),
            method=r.get("method", "notification"),
            fire_at=compute_fire_at(event_date, int(r.get("offset_minutes", 1440))),
        )
    return {"entry": entry, "event": event}
