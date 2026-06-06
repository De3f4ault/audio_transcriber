"""Migration 011: Add `strategy` column to `staging_cart` and `job_queue` tables.

Safe to run multiple times (idempotent via column-existence guards).
"""

from __future__ import annotations

import sqlite3

from audiobench.core.logger_factory import get_logger

logger = get_logger("storage.migrations.011")


def _column_exists(cursor: sqlite3.Cursor, table: str, column: str) -> bool:
    cursor.execute(f"PRAGMA table_info({table})")
    columns = [row[1] for row in cursor.fetchall()]
    return column in columns


def migrate(db_path: str) -> None:
    """Run the migration on the given SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        if not _column_exists(cursor, "staging_cart", "strategy"):
            cursor.execute("ALTER TABLE staging_cart ADD COLUMN strategy VARCHAR(64) DEFAULT 'batch'")
            logger.info("Added 'strategy' column to staging_cart")
        else:
            logger.info("'strategy' column in staging_cart already exists — skipping")

        if not _column_exists(cursor, "job_queue", "strategy"):
            cursor.execute("ALTER TABLE job_queue ADD COLUMN strategy VARCHAR(64)")
            logger.info("Added 'strategy' column to job_queue")
        else:
            logger.info("'strategy' column in job_queue already exists — skipping")

        conn.commit()
        logger.info("Migration 011 completed successfully")

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
