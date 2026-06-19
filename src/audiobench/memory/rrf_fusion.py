"""Reciprocal Rank Fusion (RRF) for blending multi-stream retrieval results."""

from __future__ import annotations

import heapq
from dataclasses import dataclass

from audiobench.memory.retrieval_streams import SegmentHit

K: int = 60  # RRF constant — must equal 60 per spec


@dataclass(frozen=True)
class FusedResult:
    """A segment hit produced by fusing scores from multiple retrieval streams."""

    segment_id: int
    start_time: float
    end_time: float
    text: str
    rrf_score: float
    source_file: str = ""
    # Tuple of (stream_name, rank) pairs so callers can see which streams contributed
    stream_contributions: tuple[tuple[str, int], ...] = ()


def rrf_merge(
    fts_hits: list[SegmentHit],
    dense_hits: list[SegmentHit],
    colbert_hits: list[SegmentHit],
    top_n: int = 10,
) -> list[FusedResult]:
    """Fuse three ranked lists via RRF.

    RRF score for a document d = sum over streams of 1 / (K + rank(d, stream))
    where rank is 1-indexed.  Documents appearing in more streams accumulate
    higher scores, rewarding cross-stream agreement.

    Uses heapq.nlargest so the time complexity is O(N log top_n) rather than
    O(N log N) for a full sort — important when N is large.
    """
    if not fts_hits and not dense_hits and not colbert_hits:
        return []

    # Map segment_id → {stream: rank, ...} and one representative SegmentHit per id
    contributions: dict[int, dict[str, int]] = {}
    hit_index: dict[int, SegmentHit] = {}

    for stream_name, hits in (
        ("fts5", fts_hits),
        ("dense", dense_hits),
        ("colbert", colbert_hits),
    ):
        for rank, hit in enumerate(hits, start=1):
            sid = hit.segment_id
            if sid not in contributions:
                contributions[sid] = {}
                hit_index[sid] = hit
            contributions[sid][stream_name] = rank

    # Compute RRF scores
    scores: dict[int, float] = {}
    for sid, stream_ranks in contributions.items():
        scores[sid] = sum(1.0 / (K + r) for r in stream_ranks.values())

    # Use a min-heap via heapq.nlargest — O(N log top_n)
    top_ids = heapq.nlargest(top_n, scores, key=lambda sid: scores[sid])

    results: list[FusedResult] = []
    for sid in top_ids:
        hit = hit_index[sid]
        contribs = contributions[sid]
        results.append(
            FusedResult(
                segment_id=sid,
                start_time=hit.start_time,
                end_time=hit.end_time,
                text=hit.text,
                rrf_score=scores[sid],
                source_file=hit.source_file,
                stream_contributions=tuple(sorted(contribs.items())),
            )
        )
    return results
