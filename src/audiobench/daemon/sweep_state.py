"""Sweep state — in-memory O(1) data structures for the RAG consistency sweep.

The daemon's sweep loop performs four classes of lookup during each pass:

1. Content-hash deduplication — "have we already embedded this text?"
2. Expression-indexed check  — "is this expression_id already in LanceDB?"
3. Pending-segment queue     — "which segment IDs still need vectorization?"
4. Unindexed-transcript queue — "which transcript IDs still need chunking?"

Without this module, all four are answered with SQL queries (O(log n) with
index, plus disk I/O).  With this module, (1) and (2) are O(1) hash-set
lookups in RAM, and (3)/(4) are O(1) deque pops — no SQL round-trips at all
during a sweep tick.

Usage
-----
    from audiobench.daemon.sweep_state import init_sweep_state, get_sweep_state

    # Once at daemon startup:
    state = init_sweep_state()
    state.load_from_db()

    # During each sweep tick (O(1) checks):
    if state.has_hash(h):
        ...
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field

from audiobench.core.logger_factory import get_logger

logger = get_logger("daemon.sweep_state")


@dataclass
class SweepState:
    """In-memory O(1) state store for the RAG consistency sweep.

    All mutating methods are protected by an RLock so they are safe to call
    from the background sweep thread AND from asyncio executor threads that
    handle incoming socket commands concurrently.
    """

    # O(1) content-hash dedup: contains every content_hash already in SQLite
    known_hashes: set[str] = field(default_factory=set)

    # O(1) LanceDB write optimiser: contains every expression_id already in
    # the LanceDB expressions table — lets us skip speculative deletes
    indexed_expression_ids: set[int] = field(default_factory=set)

    # O(1) segment queue: segment IDs waiting for vectorization
    pending_segment_ids: deque[int] = field(default_factory=deque)

    # O(1) transcript queue: transcript IDs waiting for chunking + indexing
    unindexed_transcript_ids: deque[int] = field(default_factory=deque)

    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    # ------------------------------------------------------------------
    # Bulk loader
    # ------------------------------------------------------------------

    def load_from_db(self) -> None:
        """Populate all structures from the current DB state.

        Call once at daemon startup, after the DB engine is ready.
        Also safe to call again to re-sync after a crash/restart.
        """
        from sqlalchemy import text as sql_text

        from audiobench.core.db_session import get_session

        with get_session() as session:
            # 1. Known content hashes
            hashes = session.execute(
                sql_text(
                    "SELECT content_hash FROM expressions "
                    "WHERE content_hash IS NOT NULL"
                )
            ).scalars().all()

            # 2. Un-vectorized segment IDs (ordered so we process oldest first)
            seg_ids = session.execute(
                sql_text(
                    "SELECT s.id FROM segments s "
                    "WHERE s.vector_indexed = 0 "
                    "ORDER BY s.id"
                )
            ).scalars().all()

            # 3. Unindexed transcript IDs
            tx_ids = session.execute(
                sql_text(
                    "SELECT id FROM transcriptions "
                    "WHERE is_indexed = 0 "
                    "ORDER BY id"
                )
            ).scalars().all()

        with self._lock:
            self.known_hashes = set(hashes)
            self.pending_segment_ids = deque(seg_ids)
            self.unindexed_transcript_ids = deque(tx_ids)

        logger.info(
            "SweepState loaded — %d known hashes | %d pending segments | "
            "%d unindexed transcripts",
            len(self.known_hashes),
            len(self.pending_segment_ids),
            len(self.unindexed_transcript_ids),
        )

    def sync_indexed_expression_ids(self) -> None:
        """Load expression IDs already present in LanceDB.

        Requires the MemoryStore to be initialised first.
        """
        try:
            from audiobench.memory.memory_store import MemoryStore

            store = MemoryStore.__new__(MemoryStore)  # reuse existing table
            from audiobench.core.settings import get_settings
            from pathlib import Path
            import lancedb

            settings = get_settings()
            lancedb_dir = Path(settings.data_dir) / "lancedb"
            db = lancedb.connect(str(lancedb_dir))
            if "expressions" in db.table_names():
                table = db.open_table("expressions")
                # Use to_arrow() for a full, unlimited scan of the table.
                # table.search() has a default 7000-row limit that would cause
                # indexed_expression_ids to be incomplete for large tables.
                ids = [
                    int(x) for x in table.to_arrow()["expression_id"].to_pylist()
                ]
                with self._lock:
                    self.indexed_expression_ids = set(ids)
                logger.info(
                    "SweepState: %d expression IDs loaded from LanceDB",
                    len(self.indexed_expression_ids),
                )
        except Exception as exc:
            logger.warning("SweepState: could not load indexed_expression_ids: %s", exc)

    # ------------------------------------------------------------------
    # Hash set — O(1) dedup
    # ------------------------------------------------------------------

    def has_hash(self, h: str) -> bool:
        """O(1) membership check. GIL makes reads safe without lock."""
        return h in self.known_hashes

    def add_hash(self, h: str) -> None:
        """Register a newly inserted expression hash."""
        with self._lock:
            self.known_hashes.add(h)

    def add_hashes(self, hs: list[str]) -> None:
        """Register multiple hashes at once (after a batch insert)."""
        with self._lock:
            self.known_hashes.update(hs)

    # ------------------------------------------------------------------
    # Indexed expression IDs — O(1) LanceDB write optimiser
    # ------------------------------------------------------------------

    def is_expression_indexed(self, expression_id: int) -> bool:
        """O(1) check — True if expression_id is already in LanceDB."""
        return expression_id in self.indexed_expression_ids

    def mark_expression_indexed(self, expression_id: int) -> None:
        with self._lock:
            self.indexed_expression_ids.add(expression_id)

    def mark_expressions_indexed(self, ids: list[int]) -> None:
        with self._lock:
            self.indexed_expression_ids.update(ids)

    # ------------------------------------------------------------------
    # Segment deque — O(1) pop
    # ------------------------------------------------------------------

    def push_segment(self, segment_id: int) -> None:
        """Enqueue a segment ID for vectorization."""
        with self._lock:
            self.pending_segment_ids.append(segment_id)

    def push_segments(self, ids: list[int]) -> None:
        """Enqueue multiple segment IDs."""
        with self._lock:
            self.pending_segment_ids.extend(ids)

    def pop_segment_batch(self, size: int = 64) -> list[int]:
        """Pop up to *size* segment IDs. O(1) per pop."""
        with self._lock:
            n = min(size, len(self.pending_segment_ids))
            return [self.pending_segment_ids.popleft() for _ in range(n)]

    def pending_segment_count(self) -> int:
        return len(self.pending_segment_ids)

    # ------------------------------------------------------------------
    # Transcript deque — O(1) pop
    # ------------------------------------------------------------------

    def push_transcript(self, transcript_id: int) -> None:
        """Enqueue a transcript ID for chunking/indexing."""
        with self._lock:
            self.unindexed_transcript_ids.append(transcript_id)

    def push_transcripts(self, ids: list[int]) -> None:
        with self._lock:
            self.unindexed_transcript_ids.extend(ids)

    def pop_transcript_batch(self, size: int = 20) -> list[int]:
        """Pop up to *size* transcript IDs. O(1) per pop."""
        with self._lock:
            n = min(size, len(self.unindexed_transcript_ids))
            return [self.unindexed_transcript_ids.popleft() for _ in range(n)]

    def unindexed_transcript_count(self) -> int:
        return len(self.unindexed_transcript_ids)

    # ------------------------------------------------------------------
    # Re-enqueue on failure (rollback semantics)
    # ------------------------------------------------------------------

    def requeue_segments(self, ids: list[int]) -> None:
        """Put failed segment IDs back at the front of the deque."""
        with self._lock:
            self.pending_segment_ids.extendleft(reversed(ids))

    def requeue_transcripts(self, ids: list[int]) -> None:
        """Put failed transcript IDs back at the front of the deque."""
        with self._lock:
            self.unindexed_transcript_ids.extendleft(reversed(ids))


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_state: SweepState | None = None


def init_sweep_state() -> SweepState:
    """Create and return the global SweepState singleton.

    Must be called once at daemon startup before get_sweep_state().
    """
    global _state
    _state = SweepState()
    return _state


def get_sweep_state() -> SweepState:
    """Return the global SweepState singleton."""
    if _state is None:
        raise RuntimeError(
            "SweepState not initialised — call init_sweep_state() first."
        )
    return _state
