"""Migration: Create chapters table and add chapter_id foreign keys.

Adds the `chapters` table to store chapter metadata for audio files.
Adds `chapter_id` column to `segments`, `bookmarks`, and `jobs` tables.
Safe to run multiple times (idempotent via table/column-existence guards).
"""

from __future__ import annotations

import sqlite3

from audiobench.core.logger_factory import get_logger

logger = get_logger("storage.migrations.008")


def _table_exists(cursor: sqlite3.Cursor, table: str) -> bool:
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,)
    )
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
        # 1. Create chapters table
        if not _table_exists(cursor, "chapters"):
            cursor.execute("""
                CREATE TABLE chapters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    audio_file_id INTEGER NOT NULL REFERENCES audio_files(id) ON DELETE CASCADE,
                    chapter_index INTEGER NOT NULL,
                    title TEXT NOT NULL DEFAULT 'Untitled',
                    start_time REAL NOT NULL,
                    end_time REAL NOT NULL,
                    transcription_status VARCHAR(20) DEFAULT 'pending',
                    transcription_id INTEGER REFERENCES transcriptions(id) ON DELETE SET NULL,
                    summary TEXT,
                    tags TEXT DEFAULT '[]',
                    is_ghost INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE UNIQUE INDEX ix_chapters_file_index 
                ON chapters(audio_file_id, chapter_index)
            """)
            cursor.execute("""
                CREATE INDEX ix_chapters_audio_file_id 
                ON chapters(audio_file_id)
            """)
            logger.info("Created chapters table")
        else:
            logger.info("chapters table already exists — skipping")

        # 2. Add chapter_id to segments
        if _table_exists(cursor, "segments") and not _column_exists(cursor, "segments", "chapter_id"):
            cursor.execute("ALTER TABLE segments ADD COLUMN chapter_id INTEGER REFERENCES chapters(id) ON DELETE SET NULL")
            cursor.execute("CREATE INDEX ix_segments_chapter_id ON segments(chapter_id)")
            logger.info("Added chapter_id to segments")

        # 3. Add chapter_id to bookmarks
        if _table_exists(cursor, "bookmarks") and not _column_exists(cursor, "bookmarks", "chapter_id"):
            cursor.execute("ALTER TABLE bookmarks ADD COLUMN chapter_id INTEGER REFERENCES chapters(id) ON DELETE CASCADE")
            cursor.execute("CREATE INDEX ix_bookmarks_chapter_id ON bookmarks(chapter_id)")
            logger.info("Added chapter_id to bookmarks")

        # 4. Add chapter_id to jobs
        if _table_exists(cursor, "jobs") and not _column_exists(cursor, "jobs", "chapter_id"):
            cursor.execute("ALTER TABLE jobs ADD COLUMN chapter_id INTEGER REFERENCES chapters(id) ON DELETE CASCADE")
            logger.info("Added chapter_id to jobs")
            
        # 5. Add tags to audio_files
        if _table_exists(cursor, "audio_files") and not _column_exists(cursor, "audio_files", "tags"):
            cursor.execute("ALTER TABLE audio_files ADD COLUMN tags TEXT DEFAULT '[]'")
            logger.info("Added tags to audio_files")

        conn.commit()
        logger.info("Migration 008 completed successfully")

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
