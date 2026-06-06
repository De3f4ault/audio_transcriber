"""Migration runner for AudioBench SQLite database."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from audiobench.core.logger_factory import get_logger
from audiobench.exceptions import MigrationError

logger = get_logger("storage.migrate")

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def run_migrations(db_path: Path) -> None:
    """Run all pending SQL migrations idempotently."""
    logger.info("Checking pending SQL migrations for %s", db_path)

    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(str(db_path)) as conn:
        # Enable WAL mode first as specified
        conn.execute("PRAGMA journal_mode=WAL")

        # Create schema_version table if it doesn't exist
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()

        # Get applied versions
        cursor = conn.execute("SELECT version FROM schema_version")
        applied_versions = {row[0] for row in cursor.fetchall()}

        # Find all .sql files in migrations directory
        if not MIGRATIONS_DIR.exists():
            return

        sql_files = sorted(MIGRATIONS_DIR.glob("*.sql"))

        for sql_file in sql_files:
            # Extract version from filename (e.g. 001_initial.sql -> 1)
            try:
                version = int(sql_file.stem.split("_")[0])
            except ValueError:
                logger.warning(
                    "Skipping migration file with invalid name format: %s", sql_file.name
                )
                continue

            if version in applied_versions:
                continue

            logger.info("Applying migration: %s", sql_file.name)

            sql_script = sql_file.read_text(encoding="utf-8")

            try:
                # Execute script in transaction
                conn.executescript(sql_script)
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
                conn.commit()
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e).lower():
                    logger.info("Ignoring duplicate column error for %s", sql_file.name)
                    conn.rollback()
                    conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
                    conn.commit()
                else:
                    conn.rollback()
                    raise MigrationError(f"Failed to apply {sql_file.name}: {e}") from e
            except sqlite3.Error as e:
                conn.rollback()
                raise MigrationError(f"Failed to apply {sql_file.name}: {e}") from e

    logger.info("All SQL migrations applied successfully")
