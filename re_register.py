import sys

from audiobench.core.db_session import get_session
from audiobench.storage.models import TranscriptionRecord
from audiobench.storage.repository import TranscriptionRepository
from audiobench.transcribe.transcription_result import Segment, Transcript

with get_session() as session:
    # Get latest transcription
    tx = session.query(TranscriptionRecord).order_by(TranscriptionRecord.id.desc()).first()
    if not tx:
        print("No transcriptions found")
        sys.exit(1)

    print(f"Reprocessing transcription {tx.id}...")

    # Reconstruct Transcript object
    import json

    segs = []
    for s in tx.segments:
        segs.append(
            Segment(
                id=s.segment_index,
                start=s.start_time,
                end=s.end_time,
                text=s.text,
                words=[],
                speaker=s.speaker,
                avg_logprob=0.0,
                no_speech_prob=0.0,
            )
        )

    transcript = Transcript(
        text=tx.full_text,
        segments=segs,
        language=tx.language,
        language_probability=tx.language_probability or 1.0,
        duration_seconds=tx.duration_seconds,
        word_count=tx.word_count,
        file_name=tx.file_name,
        file_hash="",
        speaker_map=json.loads(tx.speaker_map) if tx.speaker_map else {},
        engine=tx.engine,
        model_name=tx.model_name,
    )

repo = TranscriptionRepository()
repo._register_expressions(tx.id, transcript, chapter_id=None)
print("Done!")
