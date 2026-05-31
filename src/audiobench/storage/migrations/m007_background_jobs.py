"""Migration: Create jobs table for background tasks.

Adds the `jobs` table to track background execution state.
Safe to run multiple times (idempotent via table-existence guard).
"""

from __future__ import annotations

import sqlite3

from audiobench.core.logger_factory import get_logger

logger = get_logger("storage.migrations.007")


def _table_exists(cursor: sqlite3.Cursor, table: str) -> bool:
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,)
    )
    return cursor.fetchone() is not None


def migrate(db_path: str) -> None:
    """Run the migration on the given SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        if not _table_exists(cursor, "jobs"):
            cursor.execute("""
                CREATE TABLE jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    command TEXT NOT NULL,
                    pid INTEGER,
                    status VARCHAR(20) DEFAULT 'running',
                    log_path VARCHAR(1024),
                    events_path VARCHAR(1024),
                    audio_file VARCHAR(1024),
                    started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    ended_at DATETIME,
                    exit_code INTEGER
                )
            """)
            logger.info("Created jobs table")
        else:
            logger.info("jobs table already exists — skipping")

        conn.commit()
        logger.info("Migration 007 completed successfully")

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
