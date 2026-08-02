"""Migration: Add backfill_attempt_count and backfill_next_attempt_at to transcriptions."""

import sqlite3


def migrate(db_path: str) -> None:
    """Add backfill_attempt_count and backfill_next_attempt_at to transcriptions table."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check if columns already exist
        cursor.execute("PRAGMA table_info(transcriptions)")
        columns = [row[1] for row in cursor.fetchall()]

        if "backfill_attempt_count" not in columns:
            cursor.execute("ALTER TABLE transcriptions ADD COLUMN backfill_attempt_count INTEGER DEFAULT 0")

        if "backfill_next_attempt_at" not in columns:
            cursor.execute("ALTER TABLE transcriptions ADD COLUMN backfill_next_attempt_at DATETIME DEFAULT NULL")

        conn.commit()

    finally:
        conn.close()
