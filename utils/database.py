"""
utils/database.py
=================

Read-only access to a local SQLite database. Used only by the SQL
reflection section.

Three safety layers:

1. The database file is opened with the read-only URI flag.
2. A SQLite authorizer denies any operation except SELECT/READ/FUNCTION/RECURSIVE.
3. A progress handler stops queries that run too many VM steps.

This makes it impossible for a generated query to change data — even
accidentally.
"""

import sqlite3
from pathlib import Path

import pandas as pd


def open_readonly_connection(database_path):
    """Open a SQLite connection that refuses any write operation."""
    uri = Path(database_path).resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.execute("PRAGMA query_only=ON")

    allowed_operations = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_RECURSIVE,
    }

    def authorize(action, *_args):
        return sqlite3.SQLITE_OK if action in allowed_operations else sqlite3.SQLITE_DENY

    connection.set_authorizer(authorize)

    # Stop queries after about two million VM steps.
    step_counter = [0]

    def progress():
        step_counter[0] += 1
        return int(step_counter[0] > 2000)

    connection.set_progress_handler(progress, 1000)
    return connection


def get_schema_text(database_path):
    """Return the CREATE TABLE statements for every user table."""
    connection = open_readonly_connection(database_path)
    try:
        rows = connection.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        return "\n".join(row[0] for row in rows if row[0])
    finally:
        connection.close()


def run_readonly_query(query, database_path, row_limit=500):
    """
    Execute one read-only statement.

    Returns query rows as a DataFrame, or a DataFrame with an ``error``
    column on failure. Pass ``row_limit=None`` to retrieve the full result.
    """
    cleaned = query.strip().removeprefix("```sql").removesuffix("```").strip()
    connection = open_readonly_connection(database_path)
    try:
        cursor = connection.execute(cleaned)
        rows = cursor.fetchall() if row_limit is None else cursor.fetchmany(row_limit + 1)
        column_names = [description[0] for description in cursor.description]
        visible_rows = rows if row_limit is None else rows[:row_limit]
        frame = pd.DataFrame(visible_rows, columns=column_names)
        frame.attrs["truncated"] = row_limit is not None and len(rows) > row_limit
        return frame
    except Exception as error:
        return pd.DataFrame({"error": [str(error)]})
    finally:
        connection.close()
