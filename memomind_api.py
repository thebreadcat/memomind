"""REST API helpers, shared mode, Lamp integration."""

import json
import logging
from functools import wraps

import requests

import memomind_db as db
import memomind_capture as capture
import memomind_search as search
from memomind_config import load_config, resolve_db_path, save_config
from memomind_models import (
    get_provider,
    NoModelFoundError,
    ollama_running,
    get_ollama_models,
    lmstudio_running,
    configure_model,
)

logger = logging.getLogger("memomind.api")

# Integration state
_state = {
    "initialized": False,
    "db_path": None,
    "user_id": None,
    "shared_db_path": None,
    "shared_key": None,
    "shared_endpoint": None,
    "integrated_mode": False,
}


def init(
    db_path: str | None = None,
    user_id: str | None = None,
    shared_db_path: str | None = None,
    shared_key: str | None = None,
    shared_endpoint: str | None = None,
    integrated_mode: bool | None = None,
):
    """Initialize MemoMind for standalone or Lamp AI integration."""
    path = resolve_db_path(db_path)
    db.init_db(path, user_id=user_id)

    _state["initialized"] = True
    _state["db_path"] = path
    _state["user_id"] = user_id
    _state["shared_db_path"] = shared_db_path
    _state["shared_key"] = shared_key or load_config().get("shared_key")
    _state["shared_endpoint"] = shared_endpoint or load_config().get("shared_endpoint")
    # True only when Lamp (or host) passes an explicit per-user db_path argument
    _state["integrated_mode"] = (
        integrated_mode if integrated_mode is not None else db_path is not None
    )

    if user_id:
        save_config({"user_name": user_id})

    logger.info("MemoMind initialized: db=%s user=%s", path, user_id)
    return _state


def is_initialized() -> bool:
    return _state["initialized"]


def get_state() -> dict:
    return dict(_state)


def validate_shared_key(key: str | None) -> bool:
    expected = _state.get("shared_key") or load_config().get("shared_key")
    if not expected:
        return False
    return key == expected


def require_shared_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        from flask import request, jsonify
        key = request.headers.get("X-MemoMind-Key")
        if not validate_shared_key(key):
            return jsonify({
                "error": "I couldn't reach the family brain — check your connection",
                "code": "SHARED_KEY_INVALID",
            }), 403
        return f(*args, **kwargs)
    return decorated


def push_to_shared(entry_id: str, author: str | None = None) -> dict:
    """Push a personal entry to the household shared brain."""
    entry = db.get_entry(entry_id)
    if not entry:
        raise ValueError("Entry not found")

    endpoint = _state.get("shared_endpoint") or load_config().get("shared_endpoint")
    key = _state.get("shared_key") or load_config().get("shared_key")
    if not endpoint or not key:
        raise ValueError("Shared brain not configured")

    entities = db.get_entities_for_entry(entry_id)
    payload = {
        "entry_id": entry_id,
        "content": entry["content"],
        "entities": [{"type": e["type"], "value": e["value"]} for e in entities],
        "author": author or _state.get("user_id") or "unknown",
        "source_instance": f"http://localhost:{load_config().get('port', 7702)}",
    }

    r = requests.post(
        f"{endpoint.rstrip('/')}/shared/receive",
        json=payload,
        headers={"X-MemoMind-Key": key, "Content-Type": "application/json"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def receive_shared_entry(payload: dict) -> dict:
    """Receive entry pushed from another instance."""
    entry = db.create_entry(
        content=payload["content"],
        entry_type="fact",
        scope="shared",
        author=payload.get("author"),
    )
    entities = payload.get("entities", [])
    if entities:
        db.store_entities(entry["id"], entities)
    return {"entry_id": entry["id"], "stored": True}


def get_facts(user_id: str | None = None, topic: str | None = None) -> list[dict]:
    """Context injection API for Lamp AI apps."""
    if topic:
        fact = db.get_fact_by_topic(topic)
        return [fact] if fact else []
    return db.list_facts()


def get_session_info() -> dict:
    from datetime import datetime, timezone
    import memomind_persona as persona

    db.get_or_create_session()
    last_date = db.get_meta("last_entry_date")
    onboarding = db.get_meta("onboarding_complete", "false") == "true"
    user_name = db.get_meta("user_name") or load_config().get("user_name")

    days_since = 0
    if last_date:
        try:
            last_dt = datetime.fromisoformat(last_date + "T00:00:00+00:00")
            days_since = (datetime.now(timezone.utc) - last_dt).days
        except (ValueError, TypeError):
            days_since = 0

    return_msg = persona.return_message_for_days(days_since)
    last_thread = db.get_last_active_thread()
    stale = db.get_stale_entries(180)
    needs_refresh = [
        {
            "entry_id": e["id"],
            "summary": "may need updating",
            "content_preview": e["content"][:80],
        }
        for e in stale[:5]
    ]

    return {
        "days_since_last": days_since,
        "last_entry_date": last_date,
        "entry_count": db.count_entries(),
        "thread_count": db.count_threads(),
        "return_message": return_msg,
        "last_thread": {
            "id": last_thread["id"],
            "title": last_thread["title"],
            "last_entry": last_thread.get("last_entry_at", "")[:10] if last_thread else None,
        } if last_thread else None,
        "needs_refresh": needs_refresh,
        "onboarding_complete": onboarding,
        "user_name": user_name,
    }


def get_status() -> dict:
    try:
        provider = get_provider()
        model_info = provider.to_dict()
        model_ok = True
    except NoModelFoundError:
        model_info = None
        model_ok = False

    from memomind_voice import status as voice_status

    available_models: dict[str, list[dict]] = {}

    if ollama_running():
        try:
            available_models["ollama"] = [
                {"name": name, "label": name} for name in get_ollama_models()
            ]
        except Exception:
            available_models["ollama"] = []

    if lmstudio_running():
        try:
            r = requests.get("http://localhost:1234/v1/models", timeout=2)
            data = r.json()
            lm_models = []
            for m in data.get("data", []):
                name = m.get("id") or m.get("name")
                if name:
                    lm_models.append({"name": name, "label": name})
            available_models["lmstudio"] = lm_models
        except Exception:
            available_models["lmstudio"] = []

    return {
        "model": model_info,
        "model_available": model_ok,
        "db_path": _state.get("db_path") or db.get_db_path(),
        "entry_count": db.count_entries(),
        "voice": voice_status(),
        "integrated_mode": _state.get("integrated_mode", False),
        "shared_configured": bool(_state.get("shared_endpoint") or load_config().get("shared_endpoint")),
        "available_models": available_models,
    }


def set_model(provider: str, model: str | None = None) -> dict:
    """
    Change the active model provider for this server process.
    Returns updated status payload.
    """
    configure_model(provider, model)
    return get_status()


# Lamp AI app registration metadata
LAMP_APP_CONFIG = {
    "name": "MemoMind",
    "description": "Your personal memory",
    "icon": "brain",
    "db_param": "user_db_path",
    "shared_db_param": "household_db_path",
    "port_offset": 2,
    "uses_brain": True,
    "child_access": True,
}
