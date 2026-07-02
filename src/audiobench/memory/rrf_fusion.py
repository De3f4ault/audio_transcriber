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
    # Segment IDs removed by temporal deduplication — stored here so the
    # Fragment Reader can still navigate to them via H/L without gaps.
    merged_segment_ids: tuple[int, ...] = ()
    # FK → transcriptions.id; needed by the Fragment Reader for adjacency
    # queries (prev/next segments in the same transcript).
    transcription_id: int = 0


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
                transcription_id=hit.transcription_id,
                stream_contributions=tuple(sorted(contribs.items())),
            )
        )
    return results


def _temporal_dedup(fused: list[FusedResult]) -> list[FusedResult]:
    """Remove overlapping or near-adjacent fragments from the same source.

    Iterates the RRF-sorted list top-to-bottom (highest score first). The
    highest-scoring representative per time window survives. Fragments that
    are deduplicated away are not discarded — their segment IDs are stashed
    in ``merged_segment_ids`` on the surviving fragment so the Fragment
    Reader can still navigate the full transcript without gaps.

    Gap is self-calibrating: 60 % of the mean segment duration in the
    result set. A corpus of 25-second segments gets a ~15 s proximity gap;
    a corpus of 60-second segments gets a ~36 s gap. Zero configuration
    required — the chunker granularity is always the ground truth.

    Two fragments from the *same* ``source_file`` are considered duplicates
    when their time windows overlap or are within ``gap`` seconds of each
    other. Fragments from *different* sources are never deduplicated against
    each other.
    """
    import dataclasses

    if len(fused) < 2:
        return fused

    durations = [fr.end_time - fr.start_time for fr in fused]
    mean_duration = sum(durations) / len(durations)
    gap = mean_duration * 0.6

    kept: list[FusedResult] = []
    # source_file → list of (start_time, end_time, index_into_kept)
    seen: dict[str, list[tuple[float, float, int]]] = {}

    for fr in fused:  # already sorted by rrf_score descending — first = best
        windows = seen.get(fr.source_file, [])
        duplicate_of: int | None = None

        for kept_start, kept_end, kept_idx in windows:
            # Overlap test with proximity gap on both sides
            if fr.start_time < (kept_end + gap) and fr.end_time > (kept_start - gap):
                duplicate_of = kept_idx
                break

        if duplicate_of is None:
            # New unique window — add to kept and register in seen
            idx = len(kept)
            seen.setdefault(fr.source_file, []).append(
                (fr.start_time, fr.end_time, idx)
            )
            kept.append(fr)
        else:
            # Duplicate — stash this segment's id on the surviving representative
            survivor = kept[duplicate_of]
            kept[duplicate_of] = dataclasses.replace(
                survivor,
                merged_segment_ids=survivor.merged_segment_ids + (fr.segment_id,),
            )

    return kept
