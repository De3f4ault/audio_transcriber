"""ForwardingSubscriber — CLI-side observability event buffer.

Instead of each CLI subprocess opening its own connection to journal.db, the
ForwardingSubscriber accumulates events in a deque and forwards batches to the
daemon via DaemonClient.log_events(). If the daemon is unreachable, it falls
back to writing directly to journal.db via write_events_conn.

Design:
  - Thread-safe: protected by a reentrant lock.
  - Bounded: _buffer is capped at BUFFER_MAX; oldest events are discarded when
    the cap is exceeded so the CLI process is never blocked.
  - Batch threshold: events are forwarded when the buffer reaches BATCH_SIZE.
    flush() forces a forward regardless of buffer size.
  - atexit-registered: flush() is called automatically on process exit.
"""

from __future__ import annotations

import atexit
import collections
import sqlite3
import threading
from typing import Any

from audiobench.core.logger_factory import get_logger
from audiobench.daemon.client import DaemonClient
from audiobench.observatory.db import get_journal_db_path, write_events_conn

logger = get_logger("observatory.forwarding_subscriber")

BATCH_SIZE: int = 50
BUFFER_MAX: int = 500


class ForwardingSubscriber:
    """Accumulate observability events and forward them to the daemon in batches.

    Drop-in replacement for ObservabilitySubscriber in CLI subprocesses.
    """

    def __init__(
        self,
        batch_size: int = BATCH_SIZE,
        buffer_max: int = BUFFER_MAX,
    ) -> None:
        self._batch_size = batch_size
        self._buffer_max = buffer_max
        self._lock = threading.RLock()
        self._buffer: collections.deque[dict[str, Any]] = collections.deque(
            maxlen=buffer_max
        )
        self._client = DaemonClient()
        atexit.register(self.flush)

    # ------------------------------------------------------------------
    # Public API (same surface as ObservabilitySubscriber)
    # ------------------------------------------------------------------

    def install(self) -> None:
        """Register this subscriber as a wildcard listener on the EventBus.

        Must be called once at CLI process startup (i.e. from ``__main__.py``).
        All events emitted by any subsystem in this process will then flow
        through ``record()`` → batched → forwarded to the daemon.

        Calling install() a second time is a no-op because EventBus.on()
        deduplicates handlers.
        """
        from audiobench.events import get_bus

        get_bus().on("*", self.record)

    def record(self, **payload: Any) -> None:
        """Buffer one event payload.  Triggers a batch forward when BATCH_SIZE
        is reached."""
        with self._lock:
            self._buffer.append(payload)
        self._maybe_flush()

    def flush(self) -> None:
        """Drain the buffer and forward all pending events to the daemon."""
        with self._lock:
            if not self._buffer:
                return
            payloads = list(self._buffer)
            self._buffer.clear()
        self._forward(payloads)

    # ------------------------------------------------------------------
    # Introspection helpers (used by tests)
    # ------------------------------------------------------------------

    def _buffer_size(self) -> int:
        with self._lock:
            return len(self._buffer)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _maybe_flush(self) -> None:
        """Forward if buffer has reached the batch threshold."""
        with self._lock:
            if len(self._buffer) < self._batch_size:
                return
            payloads = list(self._buffer)
            self._buffer.clear()
        self._forward(payloads)

    def _forward(self, payloads: list[dict[str, Any]]) -> None:
        """Try to forward payloads to the daemon; fall back to local SQLite."""
        if not payloads:
            return
        try:
            self._client.log_events(payloads)
        except Exception:
            # Daemon unreachable — write directly to journal.db
            try:
                conn = sqlite3.connect(
                    str(get_journal_db_path()),
                    check_same_thread=False,
                    isolation_level=None,
                )
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("BEGIN TRANSACTION")
                write_events_conn(conn, payloads)
                conn.execute("COMMIT")
                conn.close()
            except Exception as exc:
                logger.error("ForwardingSubscriber fallback write failed: %s", exc)
