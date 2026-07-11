"""
Tests for 2C — AutocompleteIndex HNSW fast-path.

Three invariants (per roadmap):
1. test_build_populates_index — after build() the index reports ready=True and
   item_count > 0.
2. test_lookup_returns_top_k — lookup() returns exactly k results ordered by
   cosine distance when the index has enough entries.
3. test_lookup_before_build_raises — lookup() before build() raises RuntimeError
   with the sentinel 'INDEX_NOT_READY' so the handler can return a clean error.

All tests operate against hnswlib directly — no real LanceDB, no real embedder.
MemoryStore is patched so _build_from_store uses injected vectors.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIM = 768  # must match production constant in autocomplete.py


def _make_fake_vectors(n: int) -> list[tuple[int, list[float], str]]:
    """Return n (expression_id, unit vector, content) tuples."""
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n):
        v = rng.standard_normal(_DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        rows.append((i + 1, v.tolist(), f"expression text {i + 1}"))
    return rows


def _patched_index(n_items: int = 20):
    """Return an AutocompleteIndex whose _load_vectors is patched with fake data."""
    from audiobench.daemon.autocomplete import AutocompleteIndex

    fake_rows = _make_fake_vectors(n_items)

    idx = AutocompleteIndex()
    with patch.object(idx, "_load_vectors", return_value=fake_rows):
        idx.build()
    return idx, fake_rows


# ---------------------------------------------------------------------------
# 1. build() populates the index
# ---------------------------------------------------------------------------

def test_build_populates_index():
    """
    After calling build() with at least one item, ready must be True and the
    internal item count must equal the number of injected vectors.
    """
    n = 15
    idx, fake_rows = _patched_index(n)

    assert idx.ready is True
    assert idx.item_count() == n


# ---------------------------------------------------------------------------
# 2. lookup() returns top-k ordered results
# ---------------------------------------------------------------------------

def test_lookup_returns_top_k():
    """
    lookup() must return exactly k results. Each result must have keys
    'expression_id' and 'text'. Results are ordered closest-first (i.e. the
    nearest neighbour of the query vector is at index 0).
    """
    idx, fake_rows = _patched_index(20)

    # Use the first stored vector as the query — it should be its own nearest
    # neighbour.
    query_vector = fake_rows[0][1]
    k = 5
    results = idx.lookup(query_vector, k=k)

    assert len(results) == k, f"Expected {k} results, got {len(results)}"
    for r in results:
        assert "expression_id" in r
        assert "text" in r
    # Nearest neighbour of self is self
    assert results[0]["expression_id"] == fake_rows[0][0]


# ---------------------------------------------------------------------------
# 3. lookup() before build() raises RuntimeError
# ---------------------------------------------------------------------------

def test_lookup_before_build_raises():
    """
    Calling lookup() on an un-built index must raise RuntimeError with the
    sentinel string 'INDEX_NOT_READY' so _handle_autocomplete returns a clean
    error response instead of crashing.
    """
    from audiobench.daemon.autocomplete import AutocompleteIndex

    idx = AutocompleteIndex()
    assert idx.ready is False

    with pytest.raises(RuntimeError, match="INDEX_NOT_READY"):
        idx.lookup([0.0] * _DIM, k=3)
