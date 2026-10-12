"""Persistent SQL experiments, immutable data snapshots, and audit events."""

import hashlib
import json
import sqlite3
from datetime import datetime
from io import BytesIO
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1] / "data" / "sql_reflection"


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def connect(root=ROOT):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(root / "review_log.db")
    connection.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, title TEXT, created_at TEXT)")
    connection.execute("CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, session_id TEXT, kind TEXT, created_at TEXT, payload TEXT)")
    connection.commit()
    return connection


def new_session(root=ROOT):
    identifier = uuid4().hex
    connection = connect(root)
    try:
        with connection:
            connection.execute("INSERT INTO sessions VALUES (?, ?, ?)", (identifier, "New SQL experiment", now()))
    finally:
        connection.close()
    return identifier


def list_sessions(root=ROOT):
    connection = connect(root)
    try:
        return connection.execute("SELECT id, title, created_at FROM sessions ORDER BY rowid DESC").fetchall()
    finally:
        connection.close()


def log_event(session_id, kind, payload, root=ROOT):
    connection = connect(root)
    try:
        with connection:
            cursor = connection.execute("INSERT INTO events (session_id, kind, created_at, payload) VALUES (?, ?, ?, ?)",
                                        (session_id, kind, now(), json.dumps(payload, ensure_ascii=False)))
            if kind == "request":
                connection.execute("UPDATE sessions SET title=? WHERE id=? AND title='New SQL experiment'",
                                   (payload["user_request"][:80], session_id))
            return cursor.lastrowid
    finally:
        connection.close()


def events(session_id, root=ROOT):
    connection = connect(root)
    try:
        rows = connection.execute("SELECT id, kind, created_at, payload FROM events WHERE session_id=? ORDER BY id", (session_id,)).fetchall()
        return [dict(id=i, kind=k, created_at=t, **json.loads(p)) for i, k, t, p in rows]
    finally:
        connection.close()


def snapshot_data(raw, name, root=ROOT):
    """Preserve each data version; CSV uploads become a table named uploaded_data."""
    digest = hashlib.sha256(raw).hexdigest()
    directory = Path(root) / "sources"
    directory.mkdir(parents=True, exist_ok=True)
    suffix = Path(name).suffix.lower()
    if suffix not in (".csv", ".db", ".sqlite", ".sqlite3"):
        raise ValueError("Choose a CSV or SQLite database file.")
    original = directory / (digest + suffix)
    if not original.exists():
        original.write_bytes(raw)
    database = directory / (digest + ".db")
    if suffix == ".csv" and not database.exists():
        import pandas as pd
        dataframe = pd.read_csv(BytesIO(raw))
        connection = sqlite3.connect(database)
        try:
            with connection:
                dataframe.to_sql("uploaded_data", connection, index=False, if_exists="replace")
        except Exception:
            connection.close()
            database.unlink(missing_ok=True)
            raise
        finally:
            connection.close()
    elif suffix != ".csv":
        database = original
    return {"name": name, "sha256": digest, "snapshot": str(database), "original": str(original)}
