"""AutocompleteIndex — HNSW semantic autocomplete fast-path.

Built at daemon startup from all expression vectors in LanceDB.
Refreshed incrementally when new expressions are embedded by the RAG sweep.

Design:
  - Uses hnswlib (cosine space) for O(log n) approximate nearest-neighbour
    lookup at inference time. No embedder touched during lookup — the query
    vector is provided by the caller (already embedded by the handler).
  - _load_vectors() is a separate method so tests can patch it with fake data
    without touching LanceDB or the embedder.
  - Thread-safety: the index is rebuilt atomically. The module-level singleton
    is replaced under a lock.
  - HNSW parameters (M=32, ef_construction=200, ef=100) are tuned for a corpus
    in the 10k–500k range with 768-dimensional Nomic vectors.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import hnswlib
import numpy as np

logger = logging.getLogger("audiobench.daemon.autocomplete")

# HNSW hyperparameters
_DIM: int = 768
_M: int = 32               # Number of bidirectional links per element
_EF_CONSTRUCTION: int = 200  # Controls build quality (higher = better recall)
_EF_SEARCH: int = 100      # Controls query recall (can be set after build)
_MAX_ELEMENTS: int = 1_000_000  # Upper bound on corpus size


class AutocompleteIndex:
    """In-memory HNSW index (via hnswlib) over expression summary vectors.

    Built at daemon startup from LanceDB. Refreshed incrementally as the
    RAG sweep writes new expressions.

    Takes an already-embedded query vector and instantly returns the closest N
    expressions from the corpus.
    """

    def __init__(self) -> None:
        self.ready: bool = False
        self._index: hnswlib.Index | None = None
        self._id_map: list[int] = []   # hnswlib label → expression_id
        self._text_map: list[str] = [] # hnswlib label → expression content
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Build the in-memory HNSW index from LanceDB.

        Safe to call from a thread-pool executor — acquires the instance lock
        for the swap.
        """
        logger.info("Building AutocompleteIndex from LanceDB...")
        rows = self._load_vectors()  # list of (expression_id, vector, content)

        if not rows:
            logger.warning("AutocompleteIndex: no expression vectors found — index empty.")
            self.ready = True
            return

        ids = [r[0] for r in rows]
        texts = [r[2] for r in rows]
        matrix = np.array([r[1] for r in rows], dtype=np.float32)

        # Normalise for cosine similarity (hnswlib ip space == dot product on
        # normalised vectors == cosine similarity)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix /= norms

        n = len(ids)
        index = hnswlib.Index(space="ip", dim=_DIM)
        index.init_index(
            max_elements=max(n, _MAX_ELEMENTS),
            ef_construction=_EF_CONSTRUCTION,
            M=_M,
        )
        index.set_ef(_EF_SEARCH)
        labels = list(range(n))
        index.add_items(matrix, labels)

        with self._lock:
            self._index = index
            self._id_map = ids
            self._text_map = texts
            self.ready = True

        logger.info("AutocompleteIndex ready — %d expressions indexed.", n)

    def lookup(self, query_vector: list[float], k: int = 5) -> list[dict[str, Any]]:
        """Return up to k nearest expressions for the given query vector.

        Args:
            query_vector: 768-dimensional float vector (already embedded by the
                handler; not re-encoded here).
            k: Maximum number of results to return.

        Returns:
            List of dicts with keys ``expression_id`` and ``text``, ordered
            from closest to furthest.

        Raises:
            RuntimeError: if called before :meth:`build`.
        """
        if not self.ready or self._index is None:
            raise RuntimeError("INDEX_NOT_READY")

        v = np.array(query_vector, dtype=np.float32)
        norm = np.linalg.norm(v)
        if norm > 0:
            v /= norm

        with self._lock:
            # Clamp k to the actual number of indexed items
            actual_k = min(k, self._index.get_current_count())
            if actual_k == 0:
                return []
            labels, _distances = self._index.knn_query(v, k=actual_k)
            results = [
                {
                    "expression_id": self._id_map[label],
                    "text": self._text_map[label],
                }
                for label in labels[0]
            ]

        return results

    def item_count(self) -> int:
        """Return the number of items currently in the index."""
        if self._index is None:
            return 0
        return self._index.get_current_count()

    # ------------------------------------------------------------------
    # Internal — isolated for test patching
    # ------------------------------------------------------------------

    def _load_vectors(self) -> list[tuple[int, list[float], str]]:
        """Load all (expression_id, vector, content) rows from LanceDB.

        Returns an empty list if the table is empty or unavailable.
        Override / patch in tests to inject fake data without touching LanceDB.
        """
        try:
            from audiobench.memory.memory_store import MemoryStore
            store = MemoryStore()
            rows = store.table.search().select(["expression_id", "vector", "content"]).to_list()
            return [(int(r["expression_id"]), r["vector"], r.get("content", "")) for r in rows]
        except Exception as exc:
            logger.warning("AutocompleteIndex._load_vectors failed: %s", exc)
            return []


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_index: AutocompleteIndex | None = None
_index_lock = threading.Lock()


def get_autocomplete_index() -> AutocompleteIndex:
    """Return the module-level AutocompleteIndex singleton (create if absent)."""
    global _index
    if _index is None:
        with _index_lock:
            if _index is None:
                _index = AutocompleteIndex()
    return _index
