"""
alignment_pipeline.py — Timestamp alignment for Gemini transcripts.

WHY THIS APPROACH
-----------------
Gemini returns segment text but no timestamps.  ctc-forced-aligner (ONNX)
was used previously but leaks C++ heap memory across repeated
generate_emissions() calls — the ONNX runtime accumulates transformer
activation buffers that Python's GC cannot reach.  For a 1h 30m file at
10-second windows (550 windows), the leak grows until SIGABRT (exit 134).
No amount of gc.collect() or window-size reduction fixes a C++ heap leak.

NEW APPROACH: faster-whisper tiny
---------------------------------
faster-whisper uses CTranslate2 (C++) but correctly frees memory after
each transcription call — it does not accumulate state across calls.
The tiny model (~75 MB) runs at ~20× real-time on CPU, so a 1h 30m file
completes in about 4–5 minutes.

Algorithm
---------
1. Run faster-whisper-tiny on the full audio to get N whisper segments,
   each with a real (start, end) timestamp.
2. Build a timeline by cumulative character count over both the Whisper
   segments and the Gemini segments.
3. For each Gemini segment, interpolate its position in the Whisper
   timeline using cumulative character count as a proxy for elapsed time.

The interpolation uses character count rather than word count because it
handles multilingual (Swahili + English) and punctuation-heavy text more
robustly.  It also requires no text-matching heuristics that break when
Gemini and Whisper use slightly different words.
"""

import gc
import logging
import math
from pathlib import Path
from typing import Any, Callable

from audiobench.transcribe.transcription_result import Transcript, Word

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: faster-whisper model size to use for alignment.
#: "tiny" is ~75 MB and runs at ~20× real-time on CPU.
#: "base" is ~150 MB and runs at ~10× real-time — more accurate but slower.
_ALIGN_MODEL_SIZE: str = "tiny"


# ---------------------------------------------------------------------------
# Model loading (singleton inside the worker subprocess)
# ---------------------------------------------------------------------------

def load_align_model(device: str | None = None) -> Any:
    """Load the faster-whisper model used for timestamp alignment.

    Parameters
    ----------
    device:
        'cpu', 'cuda', or None/'auto' (auto-detects CUDA → CPU).

    Returns a ``faster_whisper.WhisperModel`` instance.  The model is
    loaded from the local HuggingFace cache — no network access required.
    """
    import torch
    from faster_whisper import WhisperModel

    if device is None or device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    compute_type = "int8"   # INT8 on CPU — fastest and smallest memory footprint
    logger.info(
        "Loading faster-whisper-%s for alignment (device=%s, compute=%s)…",
        _ALIGN_MODEL_SIZE, device, compute_type,
    )
    return WhisperModel(_ALIGN_MODEL_SIZE, device=device, compute_type=compute_type)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def align_transcript(
    transcript: Transcript,
    audio_path: str | Path,
    model: Any,
    on_progress: Callable | None = None,
) -> Transcript:
    """Align Gemini segment timestamps using faster-whisper re-transcription.

    Parameters
    ----------
    transcript:
        The Gemini transcript (segments already populated, timestamps = 0).
    audio_path:
        Path to the original audio file.
    model:
        A ``faster_whisper.WhisperModel`` instance from ``load_align_model()``.
    on_progress:
        Optional callback fired for each segment after alignment.

    Returns the mutated *transcript* with updated timestamps.
    """
    if not transcript.segments:
        return transcript

    audio_path = Path(audio_path)

    # ── Step 1: re-transcribe with the tiny model to get real timestamps ──────
    logger.info(
        "Alignment (faster-whisper-%s): transcribing %s …",
        _ALIGN_MODEL_SIZE, audio_path.name,
    )
    try:
        whisper_segments = _transcribe_for_timestamps(model, audio_path, transcript)
    except Exception as exc:
        logger.error("Alignment: faster-whisper transcription failed: %s", exc)
        return transcript

    if not whisper_segments:
        logger.warning("Alignment: faster-whisper returned no segments — keeping proportional timestamps")
        return transcript

    logger.info(
        "Alignment: faster-whisper produced %d segments covering %.0f–%.0fs",
        len(whisper_segments),
        whisper_segments[0]["start"],
        whisper_segments[-1]["end"],
    )

    # ── Step 2: patch transcript duration from the whisper timeline ───────────
    if whisper_segments:
        whisper_end = whisper_segments[-1]["end"]
        if transcript.duration_seconds == 0.0 or abs(transcript.duration_seconds - whisper_end) > 5.0:
            transcript.duration_seconds = whisper_end

    # ── Step 3: map Gemini segments → Whisper timestamps ─────────────────────
    _map_timestamps(transcript, whisper_segments)

    # ── Step 4: fire progress callbacks ───────────────────────────────────────
    if on_progress:
        for segment in transcript.segments:
            on_progress(segment)

    gc.collect()
    return transcript


# ---------------------------------------------------------------------------
# Internal: faster-whisper transcription pass
# ---------------------------------------------------------------------------

def _transcribe_for_timestamps(
    model: Any,
    audio_path: Path,
    transcript: Transcript,
) -> list[dict]:
    """Run faster-whisper on the full audio and return a list of segment dicts.

    Each dict has keys: ``start``, ``end``, ``text``, ``char_count``.

    We transcribe with ``word_timestamps=False`` (segment-level only) because
    we only need approximate timestamps.  VAD filtering is enabled so silent
    gaps don't inflate the character-time mapping.
    """
    # Infer the primary language from the Gemini transcript
    lang = transcript.language if transcript.language else None

    segments_iter, info = model.transcribe(
        str(audio_path),
        language=lang,
        task="transcribe",
        word_timestamps=False,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 500,
            "threshold": 0.3,
        },
        beam_size=1,           # fastest beam; accuracy is sufficient for alignment
        best_of=1,
        temperature=0.0,
    )

    result = []
    for seg in segments_iter:
        result.append({
            "start": float(seg.start),
            "end":   float(seg.end),
            "text":  seg.text.strip(),
            "char_count": len(seg.text.strip()),
        })

    return result


# ---------------------------------------------------------------------------
# Internal: cumulative character-count interpolation
# ---------------------------------------------------------------------------

def _map_timestamps(
    transcript: Transcript,
    whisper_segments: list[dict],
) -> None:
    """Map Gemini segment timestamps from the Whisper timeline.

    Strategy: cumulative character count as a proxy for elapsed time.

    1. Build a list of (cumulative_chars, time) anchors from the Whisper
       segments.  Cumulative chars increase monotonically; time is the
       segment *start* time.
    2. For each Gemini segment, compute where its cumulative character
       position falls in that anchor list and linearly interpolate the
       corresponding time.

    This works because:
    - Speech rate is roughly constant in talking-head / lecture audio.
    - Character count per second is approximately uniform for the same
      speaker and language.
    - We only need ±10–30 second accuracy — CTC forced alignment gives
      ±0.1s but at the cost of OOM crashes for long files.

    Handles multilingual content (Swahili + English) gracefully because
    character count is language-agnostic.
    """
    if not whisper_segments or not transcript.segments:
        return

    # Build cumulative char anchors from the Whisper side.
    # Each anchor: (cumulative_chars_at_this_segment_start, start_sec, end_sec)
    w_anchors: list[tuple[float, float, float]] = []
    cum = 0.0
    for seg in whisper_segments:
        w_anchors.append((cum, seg["start"], seg["end"]))
        cum += max(1, seg["char_count"])
    w_total_chars = cum

    # Build cumulative char positions for Gemini segments
    g_total_chars = sum(max(1, len(s.text)) for s in transcript.segments)

    if g_total_chars == 0 or w_total_chars == 0:
        return

    cum_g = 0.0
    for seg in transcript.segments:
        n_chars = max(1, len(seg.text))

        # Fractional position of this Gemini segment in [0, 1]
        g_frac_start = cum_g / g_total_chars
        g_frac_end   = (cum_g + n_chars) / g_total_chars

        # Map to absolute position in Whisper char timeline
        w_char_start = g_frac_start * w_total_chars
        w_char_end   = g_frac_end   * w_total_chars

        seg.start = _interpolate_time(w_anchors, w_char_start, whisper_segments[-1]["end"])
        seg.end   = _interpolate_time(w_anchors, w_char_end,   whisper_segments[-1]["end"])

        # Safety: end must be > start
        if seg.end <= seg.start:
            seg.end = seg.start + 0.5

        cum_g += n_chars


def _interpolate_time(
    anchors: list[tuple[float, float, float]],
    target_chars: float,
    max_time: float,
) -> float:
    """Linearly interpolate a time value from the Whisper char-time anchors.

    Parameters
    ----------
    anchors:
        Sorted list of (cumulative_chars, seg_start_sec, seg_end_sec).
    target_chars:
        The cumulative character position to look up.
    max_time:
        Clip result to this upper bound (last whisper segment end time).
    """
    if not anchors:
        return 0.0

    # Clamp below
    if target_chars <= anchors[0][0]:
        return anchors[0][1]

    # Clamp above
    if target_chars >= anchors[-1][0]:
        return min(max_time, anchors[-1][2])

    # Binary search for the surrounding pair of anchors
    lo, hi = 0, len(anchors) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if anchors[mid][0] <= target_chars:
            lo = mid
        else:
            hi = mid

    lo_chars, lo_start, lo_end = anchors[lo]
    hi_chars, hi_start, hi_end = anchors[hi]

    span = hi_chars - lo_chars
    if span == 0:
        return lo_start

    frac = (target_chars - lo_chars) / span
    # Interpolate between the end of the lo segment and start of hi segment
    return lo_end + frac * (hi_start - lo_end)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _estimate_initial_timestamps(transcript: Transcript) -> None:
    """Evenly distribute segment timestamps proportional to character count.

    Only runs when the engine returned all-zero timestamps (Gemini, etc.).
    Guards against a zero duration so it never silently produces 0:00 for all.
    Must be called AFTER transcript.duration_seconds has been set correctly.

    This provides an instant fallback before alignment runs, so the DB always
    has non-zero timestamps even if the alignment subprocess crashes.
    """
    needs_estimation = all(
        s.start == 0.0 and s.end == 0.0 for s in transcript.segments
    )
    if not needs_estimation:
        return

    duration = transcript.duration_seconds
    if duration <= 0.0:
        logger.warning(
            "Alignment: _estimate_initial_timestamps called with duration=%.1f "
            "— skipping estimation, segments will have zero timestamps.",
            duration,
        )
        return

    total_chars = sum(len(s.text) for s in transcript.segments)
    if total_chars == 0:
        return

    current_time = 0.0
    for segment in transcript.segments:
        char_count   = len(segment.text)
        seg_duration = (char_count / total_chars) * duration
        segment.start = current_time
        segment.end   = min(current_time + seg_duration, duration)
        current_time += seg_duration
