"""Migration: Add `is_indexed` column to `transcriptions`.

Safe to run multiple times (idempotent via table/column-existence guards).
"""

from __future__ import annotations

import sqlite3

from audiobench.core.logger_factory import get_logger

logger = get_logger("storage.migrations.009")


def _table_exists(cursor: sqlite3.Cursor, table: str) -> bool:
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
    return cursor.fetchone() is not None


def _column_exists(cursor: sqlite3.Cursor, table: str, column: str) -> bool:
    cursor.execute(f"PRAGMA table_info({table})")
    columns = [row[1] for row in cursor.fetchall()]
    return column in columns


def migrate(db_path: str) -> None:
    """Run the migration on the given SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        if _table_exists(cursor, "transcriptions") and not _column_exists(
            cursor, "transcriptions", "is_indexed"
        ):
            cursor.execute("ALTER TABLE transcriptions ADD COLUMN is_indexed INTEGER DEFAULT 0")
            cursor.execute(
                "CREATE INDEX ix_transcriptions_is_indexed ON transcriptions(is_indexed)"
            )
            logger.info("Added is_indexed to transcriptions")
        else:
            logger.info("is_indexed already exists on transcriptions — skipping")

        conn.commit()
        logger.info("Migration 009 completed successfully")

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
