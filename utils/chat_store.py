"""
utils/chat_store.py
===================

Persistent chat storage in a local SQLite file.

Two tables:

* conversations: (id, title)
* chat_messages: (id, thread_id, role, content, model_content)

`content` is what the user sees. `model_content` is what was sent to
the model; the two differ when documents were attached.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from uuid import uuid4

CHAT_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "chat_history.db"


def open_connection(database_path=CHAT_DB_PATH):
    """Open the chat database, creating the tables if needed."""
    Path(database_path).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS conversations "
        "(id TEXT PRIMARY KEY, title TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS chat_messages ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  thread_id TEXT NOT NULL,"
        "  role TEXT NOT NULL,"
        "  content TEXT NOT NULL,"
        "  model_content TEXT NOT NULL"
        ")"
    )
    columns = {row[1] for row in connection.execute("PRAGMA table_info(chat_messages)")}
    if "created_at" not in columns:
        connection.execute("ALTER TABLE chat_messages ADD COLUMN created_at TEXT")
    connection.commit()
    return connection


def create_conversation(database_path=CHAT_DB_PATH):
    """Insert a new conversation and return its 32-character id."""
    new_id = uuid4().hex
    connection = open_connection(database_path)
    try:
        with connection:
            connection.execute(
                "INSERT INTO conversations VALUES (?, ?)",
                (new_id, "New conversation"),
            )
    finally:
        connection.close()
    return new_id


def list_conversations(database_path=CHAT_DB_PATH):
    """Return saved conversations newest-first as (id, title) tuples."""
    connection = open_connection(database_path)
    try:
        return connection.execute(
            "SELECT id, title FROM conversations ORDER BY rowid DESC"
        ).fetchall()
    finally:
        connection.close()


def load_messages(thread_id, database_path=CHAT_DB_PATH):
    """Return the full ordered transcript of one conversation."""
    connection = open_connection(database_path)
    try:
        rows = connection.execute(
            "SELECT role, content, model_content, created_at FROM chat_messages "
            "WHERE thread_id = ? ORDER BY id",
            (thread_id,),
        ).fetchall()
        return [
            {"role": role, "content": content, "model_content": model_content,
             "created_at": created_at}
            for role, content, model_content, created_at in rows
        ]
    finally:
        connection.close()


def save_turn(thread_id, question, model_content, answer,
              database_path=CHAT_DB_PATH, *, question_timestamp=None,
              answer_timestamp=None):
    """
    Save one user question and one assistant answer atomically.

    The first question of a conversation also becomes its title.
    """
    question_timestamp = question_timestamp or datetime.now().astimezone().isoformat(timespec="seconds")
    answer_timestamp = answer_timestamp or datetime.now().astimezone().isoformat(timespec="seconds")
    connection = open_connection(database_path)
    try:
        with connection:
            exists = connection.execute(
                "SELECT 1 FROM conversations WHERE id = ?", (thread_id,)
            ).fetchone()
            if not exists:
                raise ValueError("Conversation does not exist.")

            connection.executemany(
                "INSERT INTO chat_messages "
                "(thread_id, role, content, model_content, created_at) VALUES (?, ?, ?, ?, ?)",
                [
                    (thread_id, "user", question, model_content, question_timestamp),
                    (thread_id, "assistant", answer, answer, answer_timestamp),
                ],
            )
            connection.execute(
                "UPDATE conversations SET title = ? "
                "WHERE id = ? AND title = 'New conversation'",
                (question[:60], thread_id),
            )
    finally:
        connection.close()
