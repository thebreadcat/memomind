"""Config loader, env, defaults."""

import json
import os
from pathlib import Path

CONFIG_DIR = Path.home() / ".memomind"
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULTS = {
    "db_path": None,
    "model_endpoint": None,
    "model_name": None,
    "api_key": None,
    "whisper_model": "tiny",
    "chunk_size": 20,
    "max_chunks": 10,
    "consolidation_hour": 2,
    "shared_key": None,
    "shared_endpoint": None,
    "shared_mode": False,
    "allowed_authors": [],
    "user_name": None,
    "port": 7702,
    "storage_warning_gb": 2.0,
    "storage_critical_gb": 4.0,
}


def ensure_config_dir():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def load_config():
    ensure_config_dir()
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, encoding="utf-8") as f:
                data = json.load(f)
            return {**DEFAULTS, **data}
        except (json.JSONDecodeError, OSError):
            pass
    return dict(DEFAULTS)


def save_config(updates: dict):
    ensure_config_dir()
    config = load_config()
    config.update(updates)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return config


def resolve_db_path(override=None):
    if override:
        return os.path.expanduser(override)
    if os.environ.get("MEMOMIND_DB"):
        return os.path.expanduser(os.environ["MEMOMIND_DB"])
    config = load_config()
    if config.get("db_path"):
        return os.path.expanduser(config["db_path"])
    return str(CONFIG_DIR / "memomind.db")
