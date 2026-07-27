"""Biometric worker — detached subprocess entry point.

Called by _schedule_biometric_pass() in the CLI paths to perform ECAPA-TDNN
voiceprint classification without blocking CLI exit or racing interpreter
shutdown against live C++ tensor kernels.

Usage (internal — do not call directly):
    python -m audiobench._biometric_worker <tx_id> <audio_path> <segment_ids_json>

Exit codes:
    0  — pass complete (or voiceprint not enrolled / audio missing — both are normal)
    1  — unexpected exception (logged to data/logs/biometric_worker.log)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path


def _setup_logging() -> logging.Logger:
    """Configure file-based logging so failures are visible after the parent exits."""
    from audiobench.core.settings import get_settings

    log_dir = get_settings().data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "biometric_worker.log"

    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    return logging.getLogger("biometric_worker")


def main() -> int:
    """Entry point.  Returns an exit code."""
    if len(sys.argv) != 4:
        # Invoked incorrectly — nothing useful to log yet; just exit.
        return 1

    tx_id_str, audio_path_raw, segment_ids_json = sys.argv[1], sys.argv[2], sys.argv[3]
    logger = _setup_logging()

    try:
        tx_id = int(tx_id_str)
        audio_path = audio_path_raw if audio_path_raw else None
        segment_ids: list[int] = json.loads(segment_ids_json)
    except (ValueError, json.JSONDecodeError) as exc:
        # Argument parsing failure — log and exit; parent already returned cleanly.
        logger.error("Bad arguments: %s — argv=%s", exc, sys.argv[1:])
        return 1

    logger.info("Biometric worker started for tx #%d (%d segments)", tx_id, len(segment_ids))

    try:
        from audiobench.security.voiceprint import (
            _load_audio,
            _load_ecapa,
            is_enrolled,
            tag_segments_batch,
        )
        import numpy as np

        if not is_enrolled():
            logger.info("Voiceprint not enrolled — skipping tx #%d", tx_id)
            return 0

        if not audio_path or not Path(audio_path).exists():
            logger.warning(
                "Audio file missing or not specified for tx #%d — skipping biometric pass",
                tx_id,
            )
            return 0

        from audiobench.core.db_engine import init_db
        from audiobench.core.db_session import get_session as db_session
        from audiobench.storage.models import SegmentRecord

        init_db()

        waveform, sr = _load_audio(Path(audio_path))
        model = _load_ecapa()

        with db_session() as s:
            segs = (
                s.query(SegmentRecord)
                .filter(SegmentRecord.id.in_(segment_ids))
                .filter(SegmentRecord.privacy_tier == 0)
                .order_by(SegmentRecord.segment_index)
                .all()
            )
            slices: list = []
            ids: list[int] = []
            for seg in segs:
                start = int(seg.start_time * sr)
                end = int(seg.end_time * sr)
                audio_slice = waveform[start:end]
                if len(audio_slice) >= int(sr * 0.5):
                    slices.append(audio_slice)
                    ids.append(seg.id)

        if ids:
            tag_segments_batch(ids, slices, sr, model=model)
            logger.info(
                "Biometric pass complete for tx #%d (%d/%d segments classified)",
                tx_id,
                len(ids),
                len(segment_ids),
            )
        else:
            logger.info(
                "Biometric pass: no classifiable segments for tx #%d (all too short or already tagged)",
                tx_id,
            )

        return 0

    except Exception as exc:
        # Named accepted cost: failure leaves segment at privacy_tier=0 with no retry.
        # The log line here is the sole visibility mechanism; it is greppable.
        # If this fires repeatedly, the right follow-up is a biometric_status column
        # + daemon sweep (see: audiobench session 2026-07-27, Track 3 discussion).
        logger.error(
            "Biometric pass FAILED for tx #%d — segment(s) remain at privacy_tier=0: %s",
            tx_id,
            exc,
            exc_info=True,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
