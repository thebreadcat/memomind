"""DB layer, FTS5, migrations."""

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

_db_path: str | None = None
_user_id: str | None = None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db(db_path: str, user_id: str | None = None):
    global _db_path, _user_id
    _db_path = db_path
    _user_id = user_id
    import os
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    with _connect() as conn:
        _migrate(conn)


def get_db_path() -> str:
    if not _db_path:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _db_path


@contextmanager
def _connect(retries: int = 3):
    for attempt in range(retries):
        try:
            conn = sqlite3.connect(get_db_path(), timeout=30)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            yield conn
            conn.commit()
            conn.close()
            return
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() and attempt < retries - 1:
                time.sleep(0.1 * (attempt + 1))
                continue
            raise


def _migrate(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS entries (
            id          TEXT PRIMARY KEY,
            content     TEXT NOT NULL,
            raw_input   TEXT,
            type        TEXT DEFAULT 'note',
            scope       TEXT DEFAULT 'personal',
            thread_id   TEXT,
            author      TEXT,
            strength    REAL DEFAULT 1.0,
            access_count INTEGER DEFAULT 0,
            last_accessed TEXT,
            pinned      INTEGER DEFAULT 0,
            staged      INTEGER DEFAULT 0,
            deleted_at  TEXT,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS entities (
            id          TEXT PRIMARY KEY,
            entry_id    TEXT NOT NULL,
            type        TEXT NOT NULL,
            value       TEXT NOT NULL,
            normalized  TEXT,
            created_at  TEXT NOT NULL,
            FOREIGN KEY (entry_id) REFERENCES entries(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_entities_normalized ON entities(normalized);
        CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);

        CREATE TABLE IF NOT EXISTS threads (
            id          TEXT PRIMARY KEY,
            title       TEXT NOT NULL,
            summary     TEXT,
            status      TEXT DEFAULT 'active',
            entry_count INTEGER DEFAULT 0,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL,
            last_entry_at TEXT
        );

        CREATE TABLE IF NOT EXISTS connections (
            id          TEXT PRIMARY KEY,
            entry_a     TEXT NOT NULL,
            entry_b     TEXT NOT NULL,
            strength    REAL DEFAULT 0.5,
            created_at  TEXT NOT NULL,
            last_reinforced TEXT,
            FOREIGN KEY (entry_a) REFERENCES entries(id) ON DELETE CASCADE,
            FOREIGN KEY (entry_b) REFERENCES entries(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS facts (
            id          TEXT PRIMARY KEY,
            topic       TEXT NOT NULL,
            summary     TEXT NOT NULL,
            entry_ids   TEXT NOT NULL,
            confidence  REAL DEFAULT 1.0,
            updated_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id          TEXT PRIMARY KEY,
            started_at  TEXT NOT NULL,
            last_active TEXT NOT NULL,
            entry_count INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS meta (
            key         TEXT PRIMARY KEY,
            value       TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS pending_confirmations (
            id          TEXT PRIMARY KEY,
            raw_input   TEXT NOT NULL,
            cleaned     TEXT NOT NULL,
            entities    TEXT NOT NULL,
            suggested_type TEXT,
            suggested_thread TEXT,
            conflict    TEXT,
            input_type  TEXT DEFAULT 'text',
            extra_data  TEXT,
            created_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS images (
            id              TEXT PRIMARY KEY,
            entry_id        TEXT NOT NULL,
            caption         TEXT,
            extracted_text  TEXT,
            full_path       TEXT NOT NULL,
            thumb_path      TEXT NOT NULL,
            micro_path      TEXT NOT NULL,
            original_format TEXT,
            original_size   INTEGER,
            full_size       INTEGER,
            width           INTEGER,
            height          INTEGER,
            created_at      TEXT NOT NULL,
            FOREIGN KEY (entry_id) REFERENCES entries(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS tasks (
            id              TEXT PRIMARY KEY,
            entry_id        TEXT NOT NULL,
            title           TEXT NOT NULL,
            notes           TEXT,
            status          TEXT DEFAULT 'todo',
            priority        TEXT DEFAULT 'medium',
            due_date        TEXT,
            completed_at    TEXT,
            recurrence      TEXT,
            thread_id       TEXT,
            deleted_at      TEXT,
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL,
            FOREIGN KEY (entry_id) REFERENCES entries(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
        CREATE INDEX IF NOT EXISTS idx_tasks_due ON tasks(due_date);

        CREATE TABLE IF NOT EXISTS events (
            id              TEXT PRIMARY KEY,
            entry_id        TEXT NOT NULL,
            title           TEXT NOT NULL,
            notes           TEXT,
            event_date      TEXT NOT NULL,
            end_date        TEXT,
            location        TEXT,
            all_day         INTEGER DEFAULT 0,
            recurrence      TEXT,
            next_occurrence TEXT,
            deleted_at      TEXT,
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL,
            FOREIGN KEY (entry_id) REFERENCES entries(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_events_date ON events(event_date);

        CREATE TABLE IF NOT EXISTS reminders (
            id              TEXT PRIMARY KEY,
            parent_id       TEXT NOT NULL,
            parent_type     TEXT NOT NULL,
            offset_minutes  INTEGER NOT NULL,
            label           TEXT,
            method          TEXT DEFAULT 'notification',
            fired           INTEGER DEFAULT 0,
            fire_at         TEXT NOT NULL,
            created_at      TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_reminders_fire ON reminders(fire_at, fired);
    """)

    # migrate pending_confirmations extra_data column
    cols = [r[1] for r in conn.execute("PRAGMA table_info(pending_confirmations)").fetchall()]
    if "extra_data" not in cols:
        conn.execute("ALTER TABLE pending_confirmations ADD COLUMN extra_data TEXT")

    # FTS5
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='entries_fts'"
    ).fetchone()
    if not row:
        conn.execute("""
            CREATE VIRTUAL TABLE entries_fts USING fts5(
                content,
                content='entries',
                content_rowid='rowid'
            )
        """)
        conn.execute("""
            CREATE TRIGGER entries_ai AFTER INSERT ON entries BEGIN
                INSERT INTO entries_fts(rowid, content) VALUES (new.rowid, new.content);
            END
        """)
        conn.execute("""
            CREATE TRIGGER entries_ad AFTER DELETE ON entries BEGIN
                INSERT INTO entries_fts(entries_fts, rowid, content)
                VALUES ('delete', old.rowid, old.content);
            END
        """)
        conn.execute("""
            CREATE TRIGGER entries_au AFTER UPDATE ON entries BEGIN
                INSERT INTO entries_fts(entries_fts, rowid, content)
                VALUES ('delete', old.rowid, old.content);
                INSERT INTO entries_fts(rowid, content) VALUES (new.rowid, new.content);
            END
        """)
        # Rebuild FTS from existing entries
        conn.execute("INSERT INTO entries_fts(entries_fts) VALUES('rebuild')")


def _row_to_dict(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    return dict(row)


# --- Meta ---

def get_meta(key: str, default=None):
    with _connect() as conn:
        row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        if row:
            return row["value"]
    return default


def set_meta(key: str, value: str):
    with _connect() as conn:
        conn.execute(
            "INSERT INTO meta (key, value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
            (key, value, now_iso()),
        )


# --- Entries ---

def create_entry(
    content: str,
    raw_input: str | None = None,
    entry_type: str = "note",
    scope: str = "personal",
    thread_id: str | None = None,
    author: str | None = None,
    staged: bool = False,
) -> dict:
    entry_id = str(uuid.uuid4())
    ts = now_iso()
    author = author or _user_id
    with _connect() as conn:
        conn.execute(
            """INSERT INTO entries
               (id, content, raw_input, type, scope, thread_id, author, staged, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (entry_id, content, raw_input, entry_type, scope, thread_id, author, int(staged), ts, ts),
        )
    set_meta("last_entry_date", ts[:10])
    if thread_id:
        increment_thread_count(thread_id)
    return get_entry(entry_id)


def get_entry(entry_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM entries WHERE id = ? AND deleted_at IS NULL", (entry_id,)
        ).fetchone()
        return _row_to_dict(row)


def list_entries(limit: int = 50, scope: str | None = None, offset: int = 0) -> list[dict]:
    with _connect() as conn:
        if scope:
            rows = conn.execute(
                """SELECT * FROM entries WHERE deleted_at IS NULL AND scope = ?
                   ORDER BY created_at DESC LIMIT ? OFFSET ?""",
                (scope, limit, offset),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM entries WHERE deleted_at IS NULL
                   ORDER BY created_at DESC LIMIT ? OFFSET ?""",
                (limit, offset),
            ).fetchall()
        return [_row_to_dict(r) for r in rows]


def update_entry(entry_id: str, **kwargs) -> dict | None:
    allowed = {"content", "type", "scope", "thread_id", "pinned", "strength", "staged"}
    updates = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
    if not updates:
        return get_entry(entry_id)
    updates["updated_at"] = now_iso()
    sets = ", ".join(f"{k} = ?" for k in updates)
    with _connect() as conn:
        conn.execute(f"UPDATE entries SET {sets} WHERE id = ?", (*updates.values(), entry_id))
    return get_entry(entry_id)


def soft_delete_entry(entry_id: str) -> bool:
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE entries SET deleted_at = ? WHERE id = ? AND deleted_at IS NULL",
            (now_iso(), entry_id),
        )
        return cur.rowcount > 0


def pin_entry(entry_id: str) -> dict | None:
    return update_entry(entry_id, pinned=1, strength=1.0)


def boost_entry_access(entry_id: str, boost: float = 1.5):
    entry = get_entry(entry_id)
    if not entry:
        return
    new_strength = min(entry["strength"] * boost, 1.0)
    with _connect() as conn:
        conn.execute(
            """UPDATE entries SET strength = ?, access_count = access_count + 1,
               last_accessed = ? WHERE id = ?""",
            (new_strength, now_iso(), entry_id),
        )


def count_entries(scope: str | None = None) -> int:
    with _connect() as conn:
        if scope:
            row = conn.execute(
                "SELECT COUNT(*) as c FROM entries WHERE deleted_at IS NULL AND scope = ?",
                (scope,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) as c FROM entries WHERE deleted_at IS NULL"
            ).fetchone()
        return row["c"]


def fts_search(query: str, limit: int = 200) -> list[dict]:
    with _connect() as conn:
        try:
            rows = conn.execute(
                """SELECT e.* FROM entries e
                   JOIN entries_fts fts ON e.rowid = fts.rowid
                   WHERE entries_fts MATCH ? AND e.deleted_at IS NULL
                   ORDER BY rank LIMIT ?""",
                (query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            # Fallback LIKE search if FTS query fails
            rows = conn.execute(
                """SELECT * FROM entries WHERE deleted_at IS NULL
                   AND content LIKE ? ORDER BY created_at DESC LIMIT ?""",
                (f"%{query}%", limit),
            ).fetchall()
        return [_row_to_dict(r) for r in rows]


def search_by_entities(normalized_values: list[str], limit: int = 50) -> list[dict]:
    if not normalized_values:
        return []
    placeholders = ",".join("?" * len(normalized_values))
    with _connect() as conn:
        rows = conn.execute(
            f"""SELECT DISTINCT e.* FROM entries e
                JOIN entities ent ON ent.entry_id = e.id
                WHERE ent.normalized IN ({placeholders}) AND e.deleted_at IS NULL
                ORDER BY e.created_at DESC LIMIT ?""",
            (*normalized_values, limit),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


def get_staged_entries() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM entries WHERE staged = 1 AND deleted_at IS NULL"
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


def get_all_active_entries() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM entries WHERE deleted_at IS NULL"
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


def find_similar_entries(content: str, limit: int = 5) -> list[dict]:
    words = [w for w in content.lower().split() if len(w) > 3][:5]
    if not words:
        return []
    results = []
    seen = set()
    for word in words:
        for entry in fts_search(word, limit=10):
            if entry["id"] not in seen:
                seen.add(entry["id"])
                results.append(entry)
    return results[:limit]


# --- Entities ---

def store_entities(entry_id: str, entities: list[dict]):
    ts = now_iso()
    with _connect() as conn:
        for ent in entities:
            normalized = ent.get("normalized") or ent["value"].lower().strip()
            conn.execute(
                """INSERT INTO entities (id, entry_id, type, value, normalized, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (str(uuid.uuid4()), entry_id, ent["type"], ent["value"], normalized, ts),
            )


def get_entities_for_entry(entry_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM entities WHERE entry_id = ?", (entry_id,)
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


# --- Threads ---

def create_thread(title: str, summary: str | None = None) -> dict:
    thread_id = str(uuid.uuid4())
    ts = now_iso()
    with _connect() as conn:
        conn.execute(
            """INSERT INTO threads (id, title, summary, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)""",
            (thread_id, title, summary, ts, ts),
        )
    return get_thread(thread_id)


def get_thread(thread_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM threads WHERE id = ?", (thread_id,)).fetchone()
        return _row_to_dict(row)


def list_threads(status: str | None = None) -> list[dict]:
    with _connect() as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM threads WHERE status = ? ORDER BY last_entry_at DESC",
                (status,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM threads ORDER BY last_entry_at DESC NULLS LAST, updated_at DESC"
            ).fetchall()
        return [_row_to_dict(r) for r in rows]


def update_thread(thread_id: str, **kwargs) -> dict | None:
    allowed = {"title", "summary", "status"}
    updates = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
    if not updates:
        return get_thread(thread_id)
    updates["updated_at"] = now_iso()
    sets = ", ".join(f"{k} = ?" for k in updates)
    with _connect() as conn:
        conn.execute(f"UPDATE threads SET {sets} WHERE id = ?", (*updates.values(), thread_id))
    return get_thread(thread_id)


def increment_thread_count(thread_id: str):
    ts = now_iso()
    with _connect() as conn:
        conn.execute(
            """UPDATE threads SET entry_count = entry_count + 1,
               last_entry_at = ?, updated_at = ? WHERE id = ?""",
            (ts, ts, thread_id),
        )


def get_thread_entries(thread_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """SELECT * FROM entries WHERE thread_id = ? AND deleted_at IS NULL
               ORDER BY created_at ASC""",
            (thread_id,),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


def count_threads() -> int:
    with _connect() as conn:
        return conn.execute("SELECT COUNT(*) as c FROM threads").fetchone()["c"]


def get_last_active_thread() -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM threads WHERE status = 'active' ORDER BY last_entry_at DESC LIMIT 1"
        ).fetchone()
        return _row_to_dict(row)


# --- Connections ---

def get_or_create_connection(entry_a: str, entry_b: str) -> dict:
    if entry_a > entry_b:
        entry_a, entry_b = entry_b, entry_a
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM connections WHERE entry_a = ? AND entry_b = ?",
            (entry_a, entry_b),
        ).fetchone()
        if row:
            return _row_to_dict(row)
        conn_id = str(uuid.uuid4())
        ts = now_iso()
        conn.execute(
            """INSERT INTO connections (id, entry_a, entry_b, created_at, last_reinforced)
               VALUES (?, ?, ?, ?, ?)""",
            (conn_id, entry_a, entry_b, ts, ts),
        )
        return {"id": conn_id, "entry_a": entry_a, "entry_b": entry_b, "strength": 0.5}


def reinforce_connection(entry_a: str, entry_b: str, amount: float = 0.1):
    conn_data = get_or_create_connection(entry_a, entry_b)
    new_strength = min(conn_data["strength"] + amount, 1.0)
    with _connect() as conn:
        conn.execute(
            "UPDATE connections SET strength = ?, last_reinforced = ? WHERE id = ?",
            (new_strength, now_iso(), conn_data["id"]),
        )


def get_connections_for_entries(entry_ids: list[str]) -> list[dict]:
    if not entry_ids:
        return []
    placeholders = ",".join("?" * len(entry_ids))
    with _connect() as conn:
        rows = conn.execute(
            f"""SELECT * FROM connections
                WHERE entry_a IN ({placeholders}) OR entry_b IN ({placeholders})""",
            (*entry_ids, *entry_ids),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


def get_neighbor_entry_ids(entry_id: str) -> list[tuple[str, float]]:
    neighbors = []
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM connections WHERE entry_a = ? OR entry_b = ?",
            (entry_id, entry_id),
        ).fetchall()
        for row in rows:
            other = row["entry_b"] if row["entry_a"] == entry_id else row["entry_a"]
            neighbors.append((other, row["strength"]))
    return neighbors


def prune_connections(threshold: float = 0.1):
    with _connect() as conn:
        conn.execute("DELETE FROM connections WHERE strength < ?", (threshold,))


def list_all_connections() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM connections").fetchall()
        return [_row_to_dict(r) for r in rows]


# --- Facts ---

def upsert_fact(topic: str, summary: str, entry_ids: list[str], confidence: float = 1.0):
    with _connect() as conn:
        existing = conn.execute(
            "SELECT id FROM facts WHERE topic = ?", (topic,)
        ).fetchone()
        ts = now_iso()
        ids_json = json.dumps(entry_ids)
        if existing:
            conn.execute(
                "UPDATE facts SET summary = ?, entry_ids = ?, confidence = ?, updated_at = ? WHERE topic = ?",
                (summary, ids_json, confidence, ts, topic),
            )
        else:
            conn.execute(
                "INSERT INTO facts (id, topic, summary, entry_ids, confidence, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), topic, summary, ids_json, confidence, ts),
            )


def list_facts() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM facts ORDER BY topic").fetchall()
        result = []
        for r in rows:
            d = _row_to_dict(r)
            d["entry_ids"] = json.loads(d["entry_ids"])
            result.append(d)
        return result


def get_fact_by_topic(topic: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM facts WHERE topic = ?", (topic,)).fetchone()
        if row:
            d = _row_to_dict(row)
            d["entry_ids"] = json.loads(d["entry_ids"])
            return d
        return None


# --- Pending confirmations ---

def save_pending_confirmation(data: dict) -> str:
    conf_id = data.get("id") or str(uuid.uuid4())
    suggested_thread = data.get("suggested_thread")
    if isinstance(suggested_thread, dict):
        suggested_thread = json.dumps(suggested_thread)
    with _connect() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO pending_confirmations
               (id, raw_input, cleaned, entities, suggested_type, suggested_thread, conflict, input_type, extra_data, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                conf_id,
                data["raw_input"],
                data["cleaned"],
                json.dumps(data.get("entities", [])),
                data.get("suggested_type"),
                suggested_thread,
                json.dumps(data.get("conflict")) if data.get("conflict") else None,
                data.get("input_type", "text"),
                json.dumps(data.get("extra_data")) if data.get("extra_data") else None,
                now_iso(),
            ),
        )
    return conf_id


def get_pending_confirmation(conf_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM pending_confirmations WHERE id = ?", (conf_id,)
        ).fetchone()
        if not row:
            return None
        d = _row_to_dict(row)
        d["entities"] = json.loads(d["entities"])
        if d.get("conflict"):
            d["conflict"] = json.loads(d["conflict"])
        if d.get("suggested_thread"):
            try:
                d["suggested_thread"] = json.loads(d["suggested_thread"])
            except (json.JSONDecodeError, TypeError):
                pass
        if d.get("extra_data"):
            try:
                d["extra_data"] = json.loads(d["extra_data"])
            except (json.JSONDecodeError, TypeError):
                pass
        return d


def delete_pending_confirmation(conf_id: str):
    with _connect() as conn:
        conn.execute("DELETE FROM pending_confirmations WHERE id = ?", (conf_id,))


# --- Session ---

def get_or_create_session() -> dict:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM sessions ORDER BY last_active DESC LIMIT 1"
        ).fetchone()
        ts = now_iso()
        if row:
            conn.execute(
                "UPDATE sessions SET last_active = ? WHERE id = ?",
                (ts, row["id"]),
            )
            return _row_to_dict(row)
        session_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO sessions (id, started_at, last_active) VALUES (?, ?, ?)",
            (session_id, ts, ts),
        )
        return {"id": session_id, "started_at": ts, "last_active": ts, "entry_count": 0}


def get_stale_entries(older_than_days: int = 180) -> list[dict]:
    from datetime import timedelta
    cutoff = (datetime.now(timezone.utc) - timedelta(days=older_than_days)).isoformat()
    with _connect() as conn:
        rows = conn.execute(
            """SELECT * FROM entries WHERE deleted_at IS NULL AND pinned = 0
               AND created_at < ? AND (last_accessed IS NULL OR last_accessed < ?)
               ORDER BY created_at ASC LIMIT 10""",
            (cutoff, cutoff),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


# --- Images ---

def create_image_record(entry_id: str, data: dict) -> dict:
    image_id = str(uuid.uuid4())
    ts = now_iso()
    with _connect() as conn:
        conn.execute(
            """INSERT INTO images
               (id, entry_id, caption, extracted_text, full_path, thumb_path, micro_path,
                original_format, original_size, full_size, width, height, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                image_id, entry_id, data.get("caption"), data.get("text"),
                data["full_path"], data["thumb_path"], data["micro_path"],
                data.get("original_format"), data.get("original_size"), data.get("full_size"),
                data.get("width"), data.get("height"), ts,
            ),
        )
    return get_image_by_entry(entry_id)


def get_image_by_entry(entry_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM images WHERE entry_id = ?", (entry_id,)).fetchone()
        return _row_to_dict(row)


# --- Tasks ---

def create_task(entry_id: str, title: str, **kwargs) -> dict:
    task_id = str(uuid.uuid4())
    ts = now_iso()
    recurrence = json.dumps(kwargs["recurrence"]) if kwargs.get("recurrence") else None
    with _connect() as conn:
        conn.execute(
            """INSERT INTO tasks
               (id, entry_id, title, notes, status, priority, due_date, recurrence, thread_id, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                task_id, entry_id, title,
                kwargs.get("notes"),
                kwargs.get("status", "todo"),
                kwargs.get("priority", "medium"),
                kwargs.get("due_date"),
                recurrence,
                kwargs.get("thread_id"),
                ts, ts,
            ),
        )
    return get_task(task_id)


def get_task(task_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM tasks WHERE id = ? AND deleted_at IS NULL", (task_id,)
        ).fetchone()
        if not row:
            return None
        d = _row_to_dict(row)
        if d.get("recurrence"):
            try:
                d["recurrence"] = json.loads(d["recurrence"])
            except json.JSONDecodeError:
                pass
        return d


def get_task_by_entry(entry_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM tasks WHERE entry_id = ? AND deleted_at IS NULL", (entry_id,)
        ).fetchone()
        return _row_to_dict(row)


def list_tasks(
    status: str | None = None,
    filter_name: str | None = None,
    sort: str = "priority",
) -> list[dict]:
    query = "SELECT * FROM tasks WHERE deleted_at IS NULL"
    params: list = []
    today = datetime.now(timezone.utc).date().isoformat()

    if status:
        query += " AND status = ?"
        params.append(status)
    if filter_name == "today":
        query += " AND due_date = ? AND status NOT IN ('done', 'abandoned')"
        params.append(today)
    elif filter_name == "upcoming":
        query += " AND due_date > ? AND status NOT IN ('done', 'abandoned')"
        params.append(today)
    elif filter_name == "done":
        query += " AND status = 'done'"

    order = {
        "priority": "CASE priority WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END, due_date ASC",
        "due_date": "due_date ASC NULLS LAST, created_at DESC",
        "created": "created_at DESC",
    }.get(sort, "created_at DESC")
    query += f" ORDER BY {order}"

    with _connect() as conn:
        rows = conn.execute(query, params).fetchall()
        result = []
        for r in rows:
            d = _row_to_dict(r)
            if d.get("recurrence"):
                try:
                    d["recurrence"] = json.loads(d["recurrence"])
                except json.JSONDecodeError:
                    pass
            result.append(d)
        return result


def update_task(task_id: str, **kwargs) -> dict | None:
    allowed = {"title", "notes", "status", "priority", "due_date", "completed_at", "recurrence", "thread_id"}
    updates = {}
    for k, v in kwargs.items():
        if k in allowed and v is not None:
            updates[k] = json.dumps(v) if k == "recurrence" and isinstance(v, dict) else v
    if not updates:
        return get_task(task_id)
    updates["updated_at"] = now_iso()
    sets = ", ".join(f"{k} = ?" for k in updates)
    with _connect() as conn:
        conn.execute(f"UPDATE tasks SET {sets} WHERE id = ?", (*updates.values(), task_id))
    return get_task(task_id)


def soft_delete_task(task_id: str) -> bool:
    task = get_task(task_id)
    if not task:
        return False
    with _connect() as conn:
        conn.execute("UPDATE tasks SET deleted_at = ? WHERE id = ?", (now_iso(), task_id))
    soft_delete_entry(task["entry_id"])
    return True


# --- Events ---

def create_event(entry_id: str, title: str, event_date: str, **kwargs) -> dict:
    event_id = str(uuid.uuid4())
    ts = now_iso()
    recurrence = json.dumps(kwargs["recurrence"]) if kwargs.get("recurrence") else None
    with _connect() as conn:
        conn.execute(
            """INSERT INTO events
               (id, entry_id, title, notes, event_date, end_date, location, all_day, recurrence, next_occurrence, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event_id, entry_id, title,
                kwargs.get("notes"), event_date, kwargs.get("end_date"),
                kwargs.get("location"), int(kwargs.get("all_day", 0)),
                recurrence, kwargs.get("next_occurrence", event_date),
                ts, ts,
            ),
        )
    return get_event(event_id)


def get_event(event_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM events WHERE id = ? AND deleted_at IS NULL", (event_id,)
        ).fetchone()
        if not row:
            return None
        d = _row_to_dict(row)
        if d.get("recurrence"):
            try:
                d["recurrence"] = json.loads(d["recurrence"])
            except json.JSONDecodeError:
                pass
        return d


def get_event_by_entry(entry_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM events WHERE entry_id = ? AND deleted_at IS NULL", (entry_id,)
        ).fetchone()
        return _row_to_dict(row)


def list_events(upcoming_only: bool = True, limit: int = 100) -> list[dict]:
    now = now_iso()
    query = "SELECT * FROM events WHERE deleted_at IS NULL"
    params: list = []
    if upcoming_only:
        query += " AND event_date >= ?"
        params.append(now)
    query += " ORDER BY event_date ASC LIMIT ?"
    params.append(limit)
    with _connect() as conn:
        rows = conn.execute(query, params).fetchall()
        return [_row_to_dict(r) for r in rows]


def list_events_past(limit: int = 50) -> list[dict]:
    now = now_iso()
    with _connect() as conn:
        rows = conn.execute(
            """SELECT * FROM events WHERE deleted_at IS NULL AND event_date < ?
               ORDER BY event_date DESC LIMIT ?""",
            (now, limit),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


def update_event(event_id: str, **kwargs) -> dict | None:
    allowed = {"title", "notes", "event_date", "end_date", "location", "all_day", "recurrence", "next_occurrence"}
    updates = {}
    for k, v in kwargs.items():
        if k in allowed and v is not None:
            if k == "recurrence" and isinstance(v, dict):
                updates[k] = json.dumps(v)
            elif k == "all_day":
                updates[k] = int(v)
            else:
                updates[k] = v
    if not updates:
        return get_event(event_id)
    updates["updated_at"] = now_iso()
    sets = ", ".join(f"{k} = ?" for k in updates)
    with _connect() as conn:
        conn.execute(f"UPDATE events SET {sets} WHERE id = ?", (*updates.values(), event_id))
    return get_event(event_id)


def soft_delete_event(event_id: str) -> bool:
    event = get_event(event_id)
    if not event:
        return False
    with _connect() as conn:
        conn.execute("UPDATE events SET deleted_at = ? WHERE id = ?", (now_iso(), event_id))
    soft_delete_entry(event["entry_id"])
    return True


# --- Reminders ---

def create_reminder(
    parent_id: str,
    parent_type: str,
    offset_minutes: int,
    fire_at: str,
    label: str | None = None,
    method: str = "notification",
) -> dict:
    reminder_id = str(uuid.uuid4())
    ts = now_iso()
    with _connect() as conn:
        conn.execute(
            """INSERT INTO reminders
               (id, parent_id, parent_type, offset_minutes, label, method, fired, fire_at, created_at)
               VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)""",
            (reminder_id, parent_id, parent_type, offset_minutes, label, method, fire_at, ts),
        )
    return get_reminder(reminder_id)


def get_reminder(reminder_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM reminders WHERE id = ?", (reminder_id,)).fetchone()
        return _row_to_dict(row)


def list_reminders_for_parent(parent_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM reminders WHERE parent_id = ? ORDER BY offset_minutes DESC",
            (parent_id,),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


def update_reminder(reminder_id: str, **kwargs) -> dict | None:
    allowed = {"offset_minutes", "label", "method", "fire_at", "fired"}
    updates = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
    if not updates:
        return get_reminder(reminder_id)
    sets = ", ".join(f"{k} = ?" for k in updates)
    with _connect() as conn:
        conn.execute(f"UPDATE reminders SET {sets} WHERE id = ?", (*updates.values(), reminder_id))
    return get_reminder(reminder_id)


def delete_reminder(reminder_id: str) -> bool:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))
        return cur.rowcount > 0


def mark_reminder_fired(reminder_id: str):
    with _connect() as conn:
        conn.execute("UPDATE reminders SET fired = 1 WHERE id = ?", (reminder_id,))


def get_due_reminders(before: str | None = None) -> list[dict]:
    before = before or now_iso()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM reminders WHERE fired = 0 AND fire_at <= ? ORDER BY fire_at ASC",
            (before,),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


def get_reminders_due_soon(minutes: int = 60) -> list[dict]:
    from datetime import timedelta
    end = (datetime.now(timezone.utc) + timedelta(minutes=minutes)).isoformat()
    now = now_iso()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM reminders WHERE fired = 0 AND fire_at > ? AND fire_at <= ?",
            (now, end),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


def enrich_entry(entry: dict) -> dict:
    """Attach type-specific metadata for search/UI."""
    if not entry:
        return entry
    t = entry.get("type")
    if t == "task":
        task = get_task_by_entry(entry["id"])
        if task:
            entry["task"] = {
                "status": task["status"],
                "due_date": task.get("due_date"),
                "priority": task.get("priority"),
                "title": task.get("title"),
            }
    elif t == "event":
        event = get_event_by_entry(entry["id"])
        if event:
            rems = list_reminders_for_parent(event["id"])
            next_rem = min((r["fire_at"] for r in rems if not r["fired"]), default=None)
            entry["event"] = {
                "event_date": event.get("event_date"),
                "location": event.get("location"),
                "next_reminder": next_rem,
                "title": event.get("title"),
            }
    elif t == "image":
        img = get_image_by_entry(entry["id"])
        if img:
            entry["image"] = {"thumb_path": img.get("thumb_path"), "caption": img.get("caption")}
    return entry
