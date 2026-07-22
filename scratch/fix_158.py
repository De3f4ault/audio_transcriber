from audiobench.storage.repository import TranscriptionRepository, get_session
from audiobench.storage.models import TranscriptionRecord, SegmentRecord
from audiobench.transcribe.transcription_result import Transcript, Segment, Word
import json

repo = TranscriptionRepository()

def main():
    print("Fetching transcript 158...")
    with get_session() as session:
        rec = session.query(TranscriptionRecord).get(158)
        if not rec:
            print("Record 158 not found")
            return
            
        segments = session.query(SegmentRecord).filter_by(transcription_id=158).order_by(SegmentRecord.segment_index).all()
        
        parsed_segments = []
        for s in segments:
            parsed_segments.append(Segment(
                id=s.segment_index,
                text=s.text,
                start=s.start_time,
                end=s.end_time,
                speaker=s.speaker,
                words=[]  # we don't strictly need words for chunking
            ))
            
        speaker_map = json.loads(rec.speaker_map) if rec.speaker_map else {}
            
        transcript = Transcript(
            segments=parsed_segments,
            language=rec.language,
            language_probability=rec.language_probability,
            duration_seconds=rec.duration_seconds,
            engine=rec.engine,
            model_name=rec.model_name,
            speaker_map=speaker_map
        )
        
    print("Registering expressions to Daemon...")
    repo._register_expressions(158, transcript, None)
    
    with get_session() as session:
        from audiobench.storage.models import ExpressionRecord
        count = session.query(ExpressionRecord).filter_by(source_id=158).count()
        print(f"Done! {count} expressions saved to DB and Daemon.")

if __name__ == "__main__":
    main()
