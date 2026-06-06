"""Migration 010: Add `command_events` and `workflows` tables.

The command_events table is the foundation of the intelligence layer —
it logs every REPL dispatch so patterns can be detected and workflows can
be captured. The workflows table stores named replayable sequences.

Safe to run multiple times (idempotent via table-existence guards).
"""

from __future__ import annotations

import sqlite3

from audiobench.core.logger_factory import get_logger

logger = get_logger("storage.migrations.010")


def _table_exists(cursor: sqlite3.Cursor, table: str) -> bool:
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
    return cursor.fetchone() is not None


def migrate(db_path: str) -> None:
    """Run the migration on the given SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        if not _table_exists(cursor, "command_events"):
            cursor.execute("""
                CREATE TABLE command_events (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    command         TEXT    NOT NULL,
                    args_json       TEXT    NOT NULL DEFAULT '[]',
                    context_file_id INTEGER,
                    context_tx_id   INTEGER,
                    duration_ms     INTEGER,
                    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                )
            """)
            cursor.execute("CREATE INDEX ix_command_events_command ON command_events(command)")
            cursor.execute("CREATE INDEX ix_command_events_file ON command_events(context_file_id)")
            cursor.execute("CREATE INDEX ix_command_events_ts ON command_events(created_at)")
            logger.info("Created command_events table")
        else:
            logger.info("command_events already exists — skipping")

        if not _table_exists(cursor, "workflows"):
            cursor.execute("""
                CREATE TABLE workflows (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    name        TEXT    NOT NULL UNIQUE,
                    description TEXT    NOT NULL DEFAULT '',
                    steps_json  TEXT    NOT NULL DEFAULT '[]',
                    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    updated_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                )
            """)
            logger.info("Created workflows table")
        else:
            logger.info("workflows already exists — skipping")

        conn.commit()
        logger.info("Migration 010 completed successfully")

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
