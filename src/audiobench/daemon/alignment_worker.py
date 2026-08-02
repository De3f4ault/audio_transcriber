"""
alignment_worker.py — Subprocess entry point for alignment.

Invoked by Transcriber._run_alignment_worker() via subprocess.run().
Also used by pipeline_recovery.py for daemon backfill.
Communicates result via DB, not stdout.
Exit code 0 = success, non-zero = failure (caller reads DB to confirm).

Two-phase design
----------------
Phase 1 (instant, crash-safe): proportional timestamps.
    Immediately writes evenly-distributed timestamps to the DB using only
    the total audio duration.  Even if the subprocess is killed later,
    users see non-zero timestamps (e.g. [0:02 → 0:08]) instead of [0:00].

Phase 2 (accurate, ~3–5 min for 1h30m): faster-whisper alignment.
    Re-transcribes the audio with the cached tiny model to get real segment
    timestamps, then maps those to the Gemini segments via character-count
    interpolation.  Overwrites Phase 1 timestamps if successful.

Duration resolution hierarchy (highest → lowest priority):
  1. AudioFileRecord.duration_seconds  — stored by AudioLoader at import time;
                                         always correct, never 0 for real files.
  2. torchaudio.info()                 — fast container-level probe, no decode.
  3. ffprobe CLI                       — robust fallback for any format.
"""
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _get_audio_file_duration(audio_file_id: int | None) -> float:
    """Priority 1: read duration from AudioFileRecord already in the DB.

    AudioLoader writes this at import time from the actual container
    metadata — it is always correct and never 0 for real audio files.
    """
    if not audio_file_id:
        return 0.0
    try:
        from audiobench.storage.models import AudioFileRecord
        from audiobench.core.db_session import get_session

        with get_session() as session:
            rec = session.query(AudioFileRecord).filter_by(id=audio_file_id).first()
            if rec and rec.duration_seconds > 0.0:
                return rec.duration_seconds
    except Exception:
        pass
    return 0.0


def _probe_file_duration(audio_path: Path) -> float:
    """Priority 2 + 3: probe the file directly when the DB value is missing."""
    # torchaudio — fast container-level metadata read (no full decode)
    try:
        import torchaudio
        info = torchaudio.info(str(audio_path))
        if info.num_frames > 0 and info.sample_rate > 0:
            return info.num_frames / info.sample_rate
    except Exception:
        pass

    # ffprobe — handles any format torchaudio can't
    try:
        import subprocess, json
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                str(audio_path),
            ],
            capture_output=True, text=True, timeout=30,
        )
        probe = json.loads(result.stdout)
        duration = float(probe["format"]["duration"])
        if duration > 0.0:
            return duration
    except Exception:
        pass

    return 0.0


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


def main():
    _configure_logging()

    if len(sys.argv) < 3:
        print("Usage: alignment_worker.py <tx_id> <audio_path>", file=sys.stderr)
        sys.exit(1)

    tx_id = int(sys.argv[1])
    audio_path = Path(sys.argv[2])

    try:
        from audiobench.storage.repository import TranscriptionRepository
        from audiobench.daemon.pipeline_recovery import _reconstruct_transcript
        from audiobench.transcribe.alignment_pipeline import (
            _estimate_initial_timestamps,
            load_align_model,
            align_transcript,
        )

        repo = TranscriptionRepository()
        data = repo.get_by_id(tx_id)
        if not data:
            logger.error("TranscriptionRecord %d not found.", tx_id)
            sys.exit(1)

        transcript = _reconstruct_transcript(data)

        # ── Resolve duration (critical for Gemini transcripts) ────────────────
        # TranscriptionRecord.duration_seconds is 0 for Gemini transcripts
        # because the API does not return segment timing.
        if transcript.duration_seconds == 0.0:
            duration = _get_audio_file_duration(data.get("audio_file_id"))
            if duration == 0.0:
                duration = _probe_file_duration(audio_path)

            if duration > 0.0:
                transcript.duration_seconds = duration
                logger.info(
                    "Duration patched: %.1fs (TranscriptionRecord had 0.0 — Gemini transcript)",
                    duration,
                )
            else:
                logger.warning(
                    "Could not determine audio duration — "
                    "alignment will produce proportional timestamps only."
                )

        # ── Phase 1: proportional timestamps (instant, crash-safe) ───────────
        # Write evenly-distributed timestamps to the DB NOW.
        # If this subprocess is killed at any point after this line, users
        # will see non-zero approximate timestamps instead of [0:00 → 0:00].
        _estimate_initial_timestamps(transcript)
        if transcript.duration_seconds > 0.0:
            repo.commit_alignment(tx_id, transcript)
            logger.info(
                "Phase 1: proportional timestamps committed for %d segments "
                "(%.0fs duration)",
                len(transcript.segments),
                transcript.duration_seconds,
            )
        else:
            logger.warning(
                "Phase 1 skipped: duration is 0 — cannot estimate proportional timestamps."
            )

        # ── Phase 2: faster-whisper alignment (accurate, ~3–5 min) ───────────
        # Re-transcribe with the cached tiny model to get real timestamps,
        # then map those to the Gemini segments.  This overwrites Phase 1.
        try:
            model = load_align_model()
            transcript = align_transcript(transcript, audio_path, model)
            repo.commit_alignment(tx_id, transcript)
            logger.info(
                "Phase 2: faster-whisper alignment committed for %d segments",
                len(transcript.segments),
            )
        except Exception as exc:
            logger.warning(
                "Phase 2 (faster-whisper alignment) failed: %s — "
                "proportional timestamps from Phase 1 are retained.",
                exc,
            )

        sys.exit(0)

    except Exception as exc:
        logger.error("Alignment worker failed for tx %d: %s", tx_id, exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
