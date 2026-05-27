#!/usr/bin/env python3
"""Flask server, routes, startup."""

import json
import logging
import os
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_file
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler

import memomind_api as api
import memomind_capture as capture
import memomind_db as db
import memomind_search as search
import memomind_brain as brain
import memomind_voice as voice
import memomind_images as images
import memomind_scheduler as sched
import memomind_tasks_events as te
from memomind_config import load_config, save_config, CONFIG_DIR
from memomind_models import NoModelFoundError
from memomind_persona import ERROR, COLD_START, IMAGES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("memomind")

app = Flask(__name__)
CORS(app)

scheduler = BackgroundScheduler()
_initialized = False


def ensure_initialized():
    global _initialized
    if not _initialized:
        user_id = os.environ.get("MEMOMIND_USER_ID")
        api.init(user_id=user_id)
        if not scheduler.running:
            schedule_consolidation()
        sched.start_reminder_scheduler()
        _initialized = True


@app.before_request
def _before_request():
    ensure_initialized()


def error_response(message: str, code: str, status: int = 400):
    return jsonify({"error": message, "code": code}), status


@app.errorhandler(Exception)
def handle_exception(e):
    logger.exception("Unhandled error: %s", e)
    if isinstance(e, NoModelFoundError):
        return error_response(ERROR["no_model"], "NO_MODEL", 503)
    if "locked" in str(e).lower():
        return error_response(ERROR["db_error"], "DB_LOCKED", 503)
    return error_response(str(e), "INTERNAL_ERROR", 500)


# --- Health & Status ---

@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "memomind"})


@app.route("/status")
def status():
    return jsonify(api.get_status())


@app.route("/session")
def session():
    return jsonify(api.get_session_info())


# --- Capture ---

@app.route("/capture", methods=["POST"])
def capture_route():
    data = request.get_json(force=True, silent=True) or {}
    raw = data.get("input", "").strip()
    if not raw:
        return error_response(ERROR["too_vague"], "EMPTY_INPUT")
    input_type = data.get("type", "text")
    try:
        result = capture.build_confirmation(raw, input_type=input_type)
        return jsonify(result)
    except NoModelFoundError:
        # Still allow capture with basic cleaning
        result = capture.build_confirmation(raw, input_type=input_type)
        return jsonify(result)


@app.route("/capture/confirm", methods=["POST"])
def capture_confirm():
    data = request.get_json(force=True, silent=True) or {}
    conf_id = data.get("confirmation_id")
    content = data.get("approved_content", "").strip()
    if not conf_id or not content:
        return error_response("confirmation_id and approved_content required", "INVALID_REQUEST")
    try:
        result = capture.confirm_capture(
            conf_id,
            content,
            thread_id=data.get("thread_id"),
            scope=data.get("scope", "personal"),
            update_entry_id=data.get("update_entry_id"),
            save_as=data.get("save_as"),
            task_data=data.get("task"),
            event_data=data.get("event"),
            reminders=data.get("reminders"),
        )
        return jsonify(result)
    except ValueError as e:
        return error_response(str(e), "NOT_FOUND", 404)


@app.route("/capture/discard", methods=["POST"])
def capture_discard():
    data = request.get_json(force=True, silent=True) or {}
    conf_id = data.get("confirmation_id")
    if not conf_id:
        return error_response("confirmation_id required", "INVALID_REQUEST")
    return jsonify(capture.discard_capture(conf_id))


# --- Search ---

@app.route("/search", methods=["POST"])
def search_route():
    data = request.get_json(force=True, silent=True) or {}
    query = data.get("query", "").strip()
    scope = data.get("scope", "personal")
    limit = data.get("limit", 100)
    stream = data.get("stream", False) or request.args.get("stream") == "1"

    if stream:
        def generate():
            for event in search.search_stream(query, scope=scope, limit=limit):
                yield f"event: {event['event']}\ndata: {json.dumps(event['data'])}\n\n"

        return Response(generate(), mimetype="text/event-stream")

    try:
        result = search.search(query, scope=scope, limit=limit)
        return jsonify(result)
    except NoModelFoundError:
        # FTS-only fallback
        candidates = search.prefilter(query, scope=scope, limit=limit)
        if not candidates:
            return jsonify({
                "answer": COLD_START["no_data"] if db.count_entries() == 0 else "I don't have anything on that yet.",
                "confidence": "none",
                "sourced_from": "your records",
                "chunks_processed": 0,
                "total_entries_searched": len(candidates),
                "supporting": [],
                "also_found": [],
                "contradictions": [],
            })


# --- Entries ---

@app.route("/entries", methods=["GET"])
def list_entries():
    limit = request.args.get("limit", 50, type=int)
    scope = request.args.get("scope")
    entries = [db.enrich_entry(e) for e in db.list_entries(limit=limit, scope=scope)]
    return jsonify({"entries": entries})


@app.route("/entries/<entry_id>", methods=["GET"])
def get_entry(entry_id):
    entry = db.get_entry(entry_id)
    if not entry:
        return error_response("Entry not found", "NOT_FOUND", 404)
    entry["entities"] = db.get_entities_for_entry(entry_id)
    return jsonify(entry)


@app.route("/entries/<entry_id>", methods=["PUT"])
def update_entry(entry_id):
    data = request.get_json(force=True, silent=True) or {}
    entry = db.update_entry(entry_id, **{k: data[k] for k in data if k in ("content", "type", "scope", "thread_id", "pinned")})
    if not entry:
        return error_response("Entry not found", "NOT_FOUND", 404)
    return jsonify(entry)


@app.route("/entries/<entry_id>", methods=["DELETE"])
def delete_entry(entry_id):
    if not db.soft_delete_entry(entry_id):
        return error_response("Entry not found", "NOT_FOUND", 404)
    return jsonify({"deleted": True})


@app.route("/entries/<entry_id>/related", methods=["GET"])
def related_entries(entry_id):
    related = brain.get_related_entries(entry_id)
    return jsonify({"related": related})


@app.route("/entries/<entry_id>/pin", methods=["POST"])
def pin_entry(entry_id):
    entry = db.pin_entry(entry_id)
    if not entry:
        return error_response("Entry not found", "NOT_FOUND", 404)
    return jsonify(entry)


@app.route("/entries/<entry_id>/share", methods=["POST"])
def share_entry(entry_id):
    try:
        result = api.push_to_shared(entry_id)
        return jsonify(result)
    except ValueError as e:
        return error_response(str(e), "SHARE_ERROR", 400)
    except Exception as e:
        return error_response(
            "I couldn't reach the family brain — check your connection",
            "SHARED_UNREACHABLE",
            502,
        )


# --- Threads ---

@app.route("/threads", methods=["GET"])
def list_threads():
    status_filter = request.args.get("status")
    return jsonify({"threads": db.list_threads(status=status_filter)})


@app.route("/threads", methods=["POST"])
def create_thread():
    data = request.get_json(force=True, silent=True) or {}
    title = data.get("title", "").strip()
    if not title:
        return error_response("title required", "INVALID_REQUEST")
    thread = db.create_thread(title, data.get("summary"))
    return jsonify(thread), 201


@app.route("/threads/<thread_id>", methods=["GET"])
def get_thread(thread_id):
    thread = db.get_thread(thread_id)
    if not thread:
        return error_response("Thread not found", "NOT_FOUND", 404)
    thread["entries"] = db.get_thread_entries(thread_id)
    return jsonify(thread)


@app.route("/threads/<thread_id>", methods=["PUT"])
def update_thread(thread_id):
    data = request.get_json(force=True, silent=True) or {}
    thread = db.update_thread(thread_id, **{k: data[k] for k in data if k in ("title", "summary", "status")})
    if not thread:
        return error_response("Thread not found", "NOT_FOUND", 404)
    return jsonify(thread)


@app.route("/threads/<thread_id>/close", methods=["POST"])
def close_thread(thread_id):
    thread = db.update_thread(thread_id, status="done")
    if not thread:
        return error_response("Thread not found", "NOT_FOUND", 404)
    return jsonify(thread)


@app.route("/threads/<thread_id>/summary", methods=["GET"])
def thread_summary(thread_id):
    thread = db.get_thread(thread_id)
    if not thread:
        return error_response("Thread not found", "NOT_FOUND", 404)
    entries = db.get_thread_entries(thread_id)
    if not thread.get("summary") and entries:
        brain.refresh_thread_summaries()
        thread = db.get_thread(thread_id)
    return jsonify({"summary": thread.get("summary", ""), "entry_count": len(entries)})


# --- Facts ---

@app.route("/facts", methods=["GET"])
def list_facts():
    return jsonify({"facts": db.list_facts()})


@app.route("/facts/<topic>", methods=["GET"])
def get_fact(topic):
    fact = db.get_fact_by_topic(topic)
    if not fact:
        return error_response("Fact not found", "NOT_FOUND", 404)
    return jsonify(fact)


@app.route("/facts/refresh", methods=["POST"])
def refresh_facts():
    brain.refresh_fact_summaries()
    return jsonify({"refreshed": True, "facts": db.list_facts()})


# --- Voice ---

@app.route("/voice/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        audio_data = request.get_data()
        filename = request.headers.get("X-Audio-Filename", "audio.webm")
    else:
        f = request.files["audio"]
        audio_data = f.read()
        filename = f.filename or "audio.webm"
    result = voice.transcribe_audio(audio_data, filename)
    return jsonify(result)


# --- Onboarding ---

@app.route("/onboarding/complete", methods=["POST"])
def complete_onboarding():
    data = request.get_json(force=True, silent=True) or {}
    name = data.get("name", "").strip()
    if name:
        db.set_meta("user_name", name)
        save_config({"user_name": name})
    db.set_meta("onboarding_complete", "true")
    return jsonify({"onboarding_complete": True})


@app.route("/onboarding/status", methods=["GET"])
def onboarding_status():
    complete = db.get_meta("onboarding_complete", "false") == "true"
    return jsonify({
        "onboarding_complete": complete,
        "user_name": db.get_meta("user_name"),
    })


# --- Shared Brain ---

@app.route("/shared/push", methods=["POST"])
@api.require_shared_key
def shared_push():
    data = request.get_json(force=True, silent=True) or {}
    result = api.receive_shared_entry(data)
    return jsonify(result), 201


@app.route("/shared/receive", methods=["POST"])
@api.require_shared_key
def shared_receive():
    data = request.get_json(force=True, silent=True) or {}
    result = api.receive_shared_entry(data)
    return jsonify(result), 201


@app.route("/shared/search", methods=["GET", "POST"])
@api.require_shared_key
def shared_search():
    if request.method == "POST":
        data = request.get_json(force=True, silent=True) or {}
        query = data.get("query", "")
    else:
        query = request.args.get("q", "")
    result = search.search(query, scope="shared")
    return jsonify(result)


@app.route("/shared/entries", methods=["GET"])
@api.require_shared_key
def shared_entries():
    return jsonify({"entries": db.list_entries(scope="shared")})


# --- Lamp Integration ---

@app.route("/lamp/config", methods=["GET"])
def lamp_config():
    return jsonify(api.LAMP_APP_CONFIG)


@app.route("/lamp/context", methods=["GET"])
def lamp_context():
    topic = request.args.get("topic")
    facts = api.get_facts(topic=topic)
    return jsonify({"facts": facts})


# --- Model selection ---

@app.route("/model", methods=["POST"])
def set_model_route():
    data = request.get_json(force=True, silent=True) or {}
    provider = data.get("provider")
    model = data.get("model")
    if not provider:
        return error_response("provider is required", "INVALID_REQUEST")
    try:
        status = api.set_model(provider, model)
        return jsonify(status)
    except ValueError as e:
        return error_response(str(e), "MODEL_ERROR")


# --- Consolidation ---

@app.route("/brain/consolidate", methods=["POST"])
def run_consolidation():
    brain.consolidate()
    return jsonify({"consolidated": True})


# --- Images ---

@app.route("/images/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return error_response("image file required", "INVALID_REQUEST")
    f = request.files["image"]
    caption_hint = request.form.get("caption", "")
    import tempfile
    suffix = Path(f.filename or "upload.jpg").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name
    entry = db.create_entry(
        content=caption_hint or f.filename or "Image",
        entry_type="image",
        staged=False,
    )
    try:
        processed = images.process_image(tmp_path, entry["id"])
        content = processed.get("caption") or processed.get("text") or caption_hint or "Image"
        if processed.get("text"):
            content = f"{content}\n{processed['text']}".strip()
        db.update_entry(entry["id"], content=content)
        db.create_image_record(entry["id"], processed)
        stats = images.get_storage_stats()
        return jsonify({
            "entry_id": entry["id"],
            "stored": True,
            "persona_message": IMAGES["extracted"] if processed.get("text") else IMAGES["saved"],
            "storage": stats,
        }), 201
    except Exception as e:
        db.soft_delete_entry(entry["id"])
        return error_response(str(e), "IMAGE_ERROR", 400)


@app.route("/images/<entry_id>/<size>")
def serve_image(entry_id, size):
    if size not in ("full", "thumb", "micro"):
        return error_response("invalid size", "INVALID_REQUEST", 404)
    path = images.get_image_path(entry_id, size)
    if not path.exists():
        return error_response("not found", "NOT_FOUND", 404)
    return send_file(path, mimetype="image/webp")


@app.route("/storage/stats")
def storage_stats():
    return jsonify(images.get_storage_stats())


# --- Tasks ---

@app.route("/tasks", methods=["GET"])
def list_tasks_route():
    status = request.args.get("status")
    filter_name = request.args.get("filter")
    sort = request.args.get("sort", "priority")
    tasks = db.list_tasks(status=status, filter_name=filter_name, sort=sort)
    for t in tasks:
        t["reminders"] = db.list_reminders_for_parent(t["id"])
    return jsonify({"tasks": tasks})


@app.route("/tasks", methods=["POST"])
def create_task_route():
    data = request.get_json(force=True, silent=True) or {}
    title = (data.get("title") or "").strip()
    if not title:
        return error_response("title required", "INVALID_REQUEST")
    result = te.create_task_from_confirmation(
        title=title,
        notes=data.get("notes"),
        priority=data.get("priority", "medium"),
        due_date=data.get("due_date"),
        recurrence=data.get("recurrence"),
        reminders=data.get("reminders"),
        thread_id=data.get("thread_id"),
    )
    return jsonify(result), 201


@app.route("/tasks/<task_id>", methods=["GET"])
def get_task_route(task_id):
    task = db.get_task(task_id)
    if not task:
        return error_response("Task not found", "NOT_FOUND", 404)
    task["reminders"] = db.list_reminders_for_parent(task_id)
    return jsonify(task)


@app.route("/tasks/<task_id>", methods=["PUT"])
def update_task_route(task_id):
    data = request.get_json(force=True, silent=True) or {}
    task = db.update_task(task_id, **{k: data[k] for k in data if k in (
        "title", "notes", "status", "priority", "due_date", "recurrence", "thread_id"
    )})
    if not task:
        return error_response("Task not found", "NOT_FOUND", 404)
    return jsonify(task)


@app.route("/tasks/<task_id>/status", methods=["POST"])
def task_status(task_id):
    data = request.get_json(force=True, silent=True) or {}
    status = data.get("status")
    if not status:
        return error_response("status required", "INVALID_REQUEST")
    from memomind_persona import TASKS
    completed_at = db.now_iso() if status == "done" else None
    task = db.update_task(task_id, status=status, completed_at=completed_at)
    msg = TASKS["completed"] if status == "done" else TASKS["saved"]
    return jsonify({"task": task, "persona_message": msg})


@app.route("/tasks/<task_id>/complete", methods=["POST"])
def task_complete(task_id):
    from memomind_persona import TASKS
    task = db.update_task(task_id, status="done", completed_at=db.now_iso())
    if not task:
        return error_response("Task not found", "NOT_FOUND", 404)
    return jsonify({"task": task, "persona_message": TASKS["completed"]})


@app.route("/tasks/<task_id>", methods=["DELETE"])
def delete_task_route(task_id):
    if not db.soft_delete_task(task_id):
        return error_response("Task not found", "NOT_FOUND", 404)
    return jsonify({"deleted": True})


# --- Events ---

@app.route("/events", methods=["GET"])
def list_events_route():
    upcoming = request.args.get("upcoming", "1") != "0"
    events = db.list_events(upcoming_only=upcoming)
    past = db.list_events_past() if request.args.get("include_past") else []
    for ev in events + past:
        ev["reminders"] = db.list_reminders_for_parent(ev["id"])
    return jsonify({"events": events, "past": past})


@app.route("/events/upcoming")
def events_upcoming():
    limit = request.args.get("limit", 10, type=int)
    events = db.list_events(upcoming_only=True, limit=limit)
    for ev in events:
        ev["reminders"] = db.list_reminders_for_parent(ev["id"])
    return jsonify({"events": events})


@app.route("/events", methods=["POST"])
def create_event_route():
    data = request.get_json(force=True, silent=True) or {}
    title = (data.get("title") or "").strip()
    event_date = data.get("event_date")
    if not title or not event_date:
        return error_response("title and event_date required", "INVALID_REQUEST")
    result = te.create_event_from_confirmation(
        title=title,
        event_date=event_date,
        end_date=data.get("end_date"),
        location=data.get("location"),
        all_day=data.get("all_day", False),
        notes=data.get("notes"),
        recurrence=data.get("recurrence"),
        reminders=data.get("reminders"),
        thread_id=data.get("thread_id"),
    )
    return jsonify(result), 201


@app.route("/events/<event_id>", methods=["GET"])
def get_event_route(event_id):
    event = db.get_event(event_id)
    if not event:
        return error_response("Event not found", "NOT_FOUND", 404)
    event["reminders"] = db.list_reminders_for_parent(event_id)
    return jsonify(event)


@app.route("/events/<event_id>", methods=["PUT"])
def update_event_route(event_id):
    data = request.get_json(force=True, silent=True) or {}
    event = db.update_event(event_id, **{k: data[k] for k in data if k in (
        "title", "notes", "event_date", "end_date", "location", "all_day", "recurrence"
    )})
    if not event:
        return error_response("Event not found", "NOT_FOUND", 404)
    return jsonify(event)


@app.route("/events/<event_id>", methods=["DELETE"])
def delete_event_route(event_id):
    if not db.soft_delete_event(event_id):
        return error_response("Event not found", "NOT_FOUND", 404)
    return jsonify({"deleted": True})


# --- Reminders ---

@app.route("/reminders/<parent_id>")
def list_reminders(parent_id):
    return jsonify({"reminders": db.list_reminders_for_parent(parent_id)})


@app.route("/reminders", methods=["POST"])
def create_reminder_route():
    data = request.get_json(force=True, silent=True) or {}
    parent_id = data.get("parent_id")
    parent_type = data.get("parent_type")
    offset = data.get("offset_minutes")
    fire_at = data.get("fire_at")
    if not all([parent_id, parent_type, offset is not None]):
        return error_response("parent_id, parent_type, offset_minutes required", "INVALID_REQUEST")
    if not fire_at:
        parent = db.get_task(parent_id) if parent_type == "task" else db.get_event(parent_id)
        due = parent.get("due_date") or parent.get("event_date") if parent else db.now_iso()
        fire_at = te.compute_fire_at(due, int(offset))
    reminder = db.create_reminder(
        parent_id, parent_type, int(offset), fire_at,
        label=data.get("label"), method=data.get("method", "notification"),
    )
    return jsonify(reminder), 201


@app.route("/reminders/<reminder_id>", methods=["PUT"])
def update_reminder_route(reminder_id):
    data = request.get_json(force=True, silent=True) or {}
    reminder = db.update_reminder(reminder_id, **{k: data[k] for k in data if k in (
        "offset_minutes", "label", "method", "fire_at"
    )})
    if not reminder:
        return error_response("Reminder not found", "NOT_FOUND", 404)
    return jsonify(reminder)


@app.route("/reminders/<reminder_id>", methods=["DELETE"])
def delete_reminder_route(reminder_id):
    if not db.delete_reminder(reminder_id):
        return error_response("Reminder not found", "NOT_FOUND", 404)
    return jsonify({"deleted": True})


@app.route("/reminders/due")
def reminders_due():
    minutes = request.args.get("minutes", 60, type=int)
    return jsonify({"reminders": db.get_reminders_due_soon(minutes)})


@app.route("/events/stream")
def events_stream():
    def generate():
        q = sched.sse_subscribe()
        try:
            yield "event: connected\ndata: {}\n\n"
            while True:
                try:
                    import queue as queue_mod
                    payload = q.get(timeout=30)
                    yield f"event: reminder\ndata: {json.dumps(payload)}\n\n"
                except queue_mod.Empty:
                    yield ": keepalive\n\n"
        finally:
            sched.sse_unsubscribe(q)

    return Response(generate(), mimetype="text/event-stream")


# --- PWA ---

@app.route("/favicon.ico")
def favicon():
    return "", 204


@app.route("/")
def index():
    html_path = Path(__file__).parent / "memomind.html"
    return send_file(html_path)


def schedule_consolidation():
    config = load_config()
    hour = config.get("consolidation_hour", 2)

    def job():
        try:
            brain.consolidate()
        except Exception as e:
            logger.error("Consolidation failed: %s", e)

    scheduler.add_job(job, "cron", hour=hour, minute=0, id="nightly_consolidation")
    scheduler.start()
    logger.info("Scheduled consolidation at hour %d", hour)


def main():
    config = load_config()
    port = int(os.environ.get("MEMOMIND_PORT", config.get("port", 7702)))
    ensure_initialized()
    logger.info("MemoMind starting on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
