"""Reminder loop, SSE push for standalone mode."""

import json
import logging
import queue
from queue import Empty
import threading
import time

import memomind_db as db

logger = logging.getLogger("memomind.scheduler")

_sse_clients: list[queue.Queue] = []
_loop_thread: threading.Thread | None = None
_running = False


def sse_subscribe() -> queue.Queue:
    q: queue.Queue = queue.Queue(maxsize=50)
    _sse_clients.append(q)
    return q


def sse_unsubscribe(q: queue.Queue):
    if q in _sse_clients:
        _sse_clients.remove(q)


def sse_push(payload: dict):
    dead = []
    for q in _sse_clients:
        try:
            q.put_nowait(payload)
        except queue.Full:
            dead.append(q)
    for q in dead:
        sse_unsubscribe(q)


def build_reminder_message(reminder: dict, parent: dict) -> str:
    base = reminder.get("label") or parent.get("title") or "Reminder"
    try:
        from memomind_db import get_fact_by_topic
        fact = get_fact_by_topic(parent.get("title", ""))
        if fact and fact.get("summary"):
            return f"{base} — {fact['summary']}"
    except Exception:
        pass
    return base


def fire_reminder(reminder: dict):
    parent_type = reminder["parent_type"]
    parent = None
    if parent_type == "task":
        parent = db.get_task(reminder["parent_id"])
    elif parent_type == "event":
        parent = db.get_event(reminder["parent_id"])
    if not parent:
        return
    message = build_reminder_message(reminder, parent)
    db.mark_reminder_fired(reminder["id"])
    sse_push({"type": "reminder", "message": message, "reminder_id": reminder["id"]})
    logger.info("Reminder fired: %s", message)


def reminder_loop():
    global _running
    _running = True
    while _running:
        try:
            due = db.get_due_reminders()
            for reminder in due:
                fire_reminder(reminder)
        except Exception as e:
            logger.error("Reminder loop error: %s", e)
        time.sleep(60)


def start_reminder_scheduler():
    global _loop_thread
    if _loop_thread and _loop_thread.is_alive():
        return
    _loop_thread = threading.Thread(target=reminder_loop, daemon=True, name="memomind-reminders")
    _loop_thread.start()
    logger.info("Reminder scheduler started")


def stop_reminder_scheduler():
    global _running
    _running = False
