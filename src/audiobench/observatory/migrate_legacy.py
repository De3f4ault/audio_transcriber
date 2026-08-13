"""Migration for legacy observatory data."""

from __future__ import annotations

import json
import sqlite3
import sys

from audiobench.core.settings import get_settings
from audiobench.observatory.db import get_journal_db_path, write_events_conn


def migrate_command_events() -> None:
    """Migrate legacy command_events from transcriptions.db to system_events in journal.db."""
    settings = get_settings()
    journal_path = get_journal_db_path()
    tx_path = settings.data_dir / "transcriptions.db"

    if not tx_path.exists():
        return

    try:
        conn = sqlite3.connect(str(journal_path), isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")

        # Check if already migrated
        cur = conn.execute("SELECT value FROM journal_meta WHERE key='command_events_migrated'")
        row = cur.fetchone()
        if row and row[0] == '1':
            conn.close()
            return

        print("[observatory] Migrating legacy command_events...", file=sys.stderr)

        tx_conn = sqlite3.connect(f"file:{tx_path}?mode=ro", uri=True)
        tx_cur = tx_conn.execute(
            "SELECT id, command, args_json, context_file_id, context_tx_id, duration_ms, created_at "
            "FROM command_events ORDER BY created_at ASC"
        )

        batch = []
        for r in tx_cur.fetchall():
            cmd = r[1]
            try:
                args = json.loads(r[2]) if r[2] else []
            except Exception:
                args = []

            meta = {
                "command": cmd,
                "args": args,
                "context_file_id": r[3],
                "context_tx_id": r[4]
            }

            payload = {
                "ts": r[6],
                "level": "INFO",
                "subsystem": "repl",
                "event_type": "command_dispatched",
                "source": "legacy_command_events",
                "metadata": meta,
                "duration_ms": r[5]
            }
            batch.append(payload)

        if batch:
            # Insert in chunks of 500
            for i in range(0, len(batch), 500):
                chunk = batch[i:i+500]
                conn.execute("BEGIN TRANSACTION")
                write_events_conn(conn, chunk)
                conn.execute("COMMIT")

        conn.execute(
            "INSERT OR REPLACE INTO journal_meta (key, value) VALUES (?, ?)",
            ("command_events_migrated", "1")
        )

        tx_conn.close()
        conn.close()
        print(f"[observatory] Migrated {len(batch)} command events.", file=sys.stderr)

    except Exception as exc:
        print(f"[observatory] Warning: Failed to migrate command_events: {exc}", file=sys.stderr)


def migrate_log_files() -> None:
    """Migrate legacy audiobench.log and worker.log to system_events in journal.db."""
    settings = get_settings()
    journal_path = get_journal_db_path()

    try:
        conn = sqlite3.connect(str(journal_path), isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")

        # Check if already migrated
        cur = conn.execute("SELECT value FROM journal_meta WHERE key='legacy_log_migrated'")
        row = cur.fetchone()
        if row and row[0] == '1':
            conn.close()
            return

        print("[observatory] Migrating legacy log files...", file=sys.stderr)

        import re
        log_pattern = re.compile(r'^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:,\d{3})?)\s*\|\s*(?P<name>audiobench\.\S+)\s*\|\s*(?P<level>\S+)\s*\|\s*(?P<message>.*)$')

        batch = []

        for log_file in ["audiobench.log", "worker.log"]:
            path = settings.data_dir / "logs" / log_file
            if not path.exists():
                continue

            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    m = log_pattern.match(line)
                    if m:
                        ts = m.group("ts").replace(",", ".")  # Make it ISO-like
                        name = m.group("name")
                        level = m.group("level")
                        message = m.group("message")

                        parts = name.split(".")
                        subsystem = parts[1] if len(parts) > 1 else "unknown"

                        batch.append({
                            "ts": ts,
                            "level": level,
                            "subsystem": subsystem,
                            "event_type": "legacy_log",
                            "message": message[:500],
                            "source": "legacy_log"
                        })

        if batch:
            for i in range(0, len(batch), 500):
                chunk = batch[i:i+500]
                conn.execute("BEGIN TRANSACTION")
                write_events_conn(conn, chunk)
                conn.execute("COMMIT")

        conn.execute(
            "INSERT OR REPLACE INTO journal_meta (key, value) VALUES (?, ?)",
            ("legacy_log_migrated", "1")
        )

        conn.close()
        print(f"[observatory] Migrated {len(batch)} legacy log lines.", file=sys.stderr)

    except Exception as exc:
        print(f"[observatory] Warning: Failed to migrate legacy log files: {exc}", file=sys.stderr)
