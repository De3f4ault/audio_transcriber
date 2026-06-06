"""Migration: Add refined_at column to transcriptions.

Adds a nullable ``refined_at`` datetime column that records when a
transcription was last cleaned by the LLM refiner.  The column is NULL
for unrefined (raw Whisper) transcriptions and populated by
``TranscriptionRepository.update_full_text()`` on first/subsequent
cleaning.

Safe to run multiple times (idempotent via column-existence guard).
"""

from __future__ import annotations

import sqlite3

from audiobench.core.logger_factory import get_logger

logger = get_logger("storage.migrations.006")


def _column_exists(cursor: sqlite3.Cursor, table: str, column: str) -> bool:
    cursor.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cursor.fetchall())


def migrate(db_path: str) -> None:
    """Run the migration on the given SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # ── transcriptions.refined_at ──
        if not _column_exists(cursor, "transcriptions", "refined_at"):
            cursor.execute("ALTER TABLE transcriptions ADD COLUMN refined_at DATETIME")
            logger.info("Added refined_at column to transcriptions")
        else:
            logger.info("refined_at column already exists — skipping")

        conn.commit()
        logger.info("Migration 006 completed successfully")

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
