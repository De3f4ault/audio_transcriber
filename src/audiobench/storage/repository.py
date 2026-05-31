"""Repository — CRUD operations for transcription data.

Provides a clean interface over SQLAlchemy for:
- Saving transcriptions with deduplication (by file hash)
- Querying transcription history
- Searching past transcriptions by text
"""

from __future__ import annotations

from datetime import UTC, datetime
import json

from sqlalchemy import desc

from audiobench.core.db_session import get_session
from audiobench.core.logger_factory import get_logger
from audiobench.storage.models import AudioFileRecord, SegmentRecord, TranscriptionRecord
from audiobench.transcribe.transcription_result import AudioMetadata, Transcript

logger = get_logger("storage.repository")


class TranscriptionRepository:
    """CRUD operations for transcription persistence."""

    def _import_to_library(self, original_path: str, move: bool = True) -> str:
        """Import an audio file into the managed data/library directory.

        If ``move=True`` (default) the original file is moved, not copied,
        making the library the single source of truth. If the move succeeds
        but a later step raises, the file is moved back before re-raising so
        the caller never loses the original.

        Also moves any .cue / .srt / .vtt sidecars alongside the audio file.
        """
        from pathlib import Path
        import shutil
        import hashlib
        from audiobench.core.settings import get_settings

        settings = get_settings()
        library_dir = settings.data_dir / "library"
        library_dir.mkdir(parents=True, exist_ok=True)

        orig = Path(original_path).absolute()
        if library_dir in orig.parents or not orig.exists():
            return str(orig)

        file_id_hash = hashlib.md5(str(orig).encode()).hexdigest()[:8]
        target_name = f"{file_id_hash}_{orig.name}"
        target_path = library_dir / target_name

        moved_pairs: list[tuple] = []  # (dest, src) for rollback

        try:
            if not target_path.exists():
                if move:
                    shutil.move(str(orig), str(target_path))
                    moved_pairs.append((target_path, orig))
                else:
                    shutil.copy2(str(orig), str(target_path))

            # Move sidecars alongside the audio file
            for ext in (".cue", ".srt", ".vtt"):
                sidecar = orig.with_suffix(ext)
                if sidecar.exists():
                    sidecar_target = target_path.with_suffix(ext)
                    if not sidecar_target.exists():
                        if move:
                            shutil.move(str(sidecar), str(sidecar_target))
                            moved_pairs.append((sidecar_target, sidecar))
                        else:
                            shutil.copy2(str(sidecar), str(sidecar_target))

            return str(target_path)

        except Exception:
            # Roll back: move each already-moved file back to its origin
            for dest, src in reversed(moved_pairs):
                try:
                    if Path(dest).exists():
                        shutil.move(str(dest), str(src))
                except Exception as rb_err:
                    logger.error("Rollback failed for %s → %s: %s", dest, src, rb_err)
            raise

    def save_transcription(
        self,
        transcript: Transcript,
        audio_metadata: AudioMetadata | None = None,
        chapter_id: int | None = None,
    ) -> int:
        """Save a transcription result to the database.

        If the same audio file (by hash) was already transcribed, the audio
        record is reused and a new transcription is linked to it.

        Args:
            transcript: The transcription result.
            audio_metadata: Source audio metadata (for dedup by hash).
            chapter_id: Optional chapter ID if this is a chapter transcription.

        Returns:
            The transcription record ID.
        """
        with get_session() as session:
            # Find or create audio file record
            audio_record = None
            chapter_record = None
            
            if chapter_id:
                from audiobench.storage.models import ChapterRecord
                chapter_record = session.query(ChapterRecord).get(chapter_id)
                if chapter_record:
                    audio_record = session.query(AudioFileRecord).get(chapter_record.audio_file_id)
            elif audio_metadata and audio_metadata.file_hash:
                audio_record = (
                    session.query(AudioFileRecord)
                    .filter_by(file_hash=audio_metadata.file_hash)
                    .first()
                )

            if audio_record is None and audio_metadata and not chapter_id:
                # Import file to library
                new_path = self._import_to_library(audio_metadata.file_path)
                
                audio_record = AudioFileRecord(
                    file_path=new_path,
                    file_name=audio_metadata.file_name,
                    file_size_bytes=audio_metadata.file_size_bytes,
                    format=audio_metadata.format,
                    duration_seconds=audio_metadata.duration_seconds,
                    sample_rate=audio_metadata.sample_rate,
                    channels=audio_metadata.channels,
                    file_hash=audio_metadata.file_hash,
                )
                session.add(audio_record)
                session.flush()  # Get the ID

                # Auto-detect chapters
                from audiobench.chapters.detector import ChapterDetector
                from pathlib import Path
                
                try:
                    detector = ChapterDetector()
                    chapters_info = detector.detect(Path(new_path))
                    if chapters_info:
                        chap_dicts = [
                            {
                                "index": c.index,
                                "title": c.title,
                                "start_time": c.start_time,
                                "end_time": c.end_time,
                                "is_ghost": c.is_ghost,
                            }
                            for c in chapters_info
                        ]
                        # ChapterRepository manages its own session, but we are inside one.
                        # Wait, ChapterRepository.save_chapters opens its own `with get_session()`.
                        # Since we are already in a transaction, it might block if using sqlite with WAL,
                        # but get_session() typically handles nested or new connections.
                        # However, AudioFileRecord might not be committed yet!
                        # We must commit first so ChapterRepository can see the audio_file_id.
                        session.commit()

                        from audiobench.storage.chapter_repository import get_chapter_repo
                        get_chapter_repo().save_chapters(audio_record.id, chap_dicts)
                        logger.info("Auto-detected and saved %d chapters for %s", len(chapters_info), audio_record.file_name)
                except Exception as e:
                    logger.warning("Failed to auto-detect chapters for %s: %s", audio_record.file_name, e)

            # Create transcription record
            tx_record = TranscriptionRecord(
                audio_file_id=audio_record.id if audio_record else None,
                source="file",
                file_name=audio_metadata.file_name if audio_metadata else "",
                full_text=transcript.text,
                language=transcript.language,
                language_probability=transcript.language_probability,
                engine=transcript.engine,
                model_name=transcript.model_name,
                duration_seconds=transcript.duration_seconds,
                word_count=transcript.word_count,
                segment_count=transcript.segment_count,
                status="completed",
                speaker_map=json.dumps(transcript.speaker_map),
            )
            session.add(tx_record)
            session.flush()

            # Save segments
            for seg in transcript.segments:
                seg_record = SegmentRecord(
                    transcription_id=tx_record.id,
                    segment_index=seg.id,
                    text=seg.text,
                    start_time=seg.start,
                    end_time=seg.end,
                    speaker=seg.speaker,
                    chapter_id=chapter_id,
                )
                session.add(seg_record)
                
            if chapter_record:
                chapter_record.transcription_id = tx_record.id
                chapter_record.transcription_status = "completed"

            session.commit()
            logger.info(
                "Saved transcription #%d (%d segments)", tx_record.id, len(transcript.segments)
            )
            return tx_record.id

    def save_live_session(self, transcript: Transcript) -> int:
        """Save a live transcription session to the database.

        Live sessions have no source audio file.

        Returns:
            The transcription record ID.
        """
        with get_session() as session:
            tx_record = TranscriptionRecord(
                audio_file_id=None,
                source="live",
                file_name="🎤 Live session",
                full_text=transcript.text,
                language=transcript.language,
                language_probability=transcript.language_probability,
                engine="faster-whisper",
                model_name=transcript.model_name if transcript.model_name else "base",
                duration_seconds=transcript.duration_seconds,
                word_count=transcript.word_count,
                segment_count=transcript.segment_count,
                status="completed",
            )
            session.add(tx_record)
            session.flush()

            for seg in transcript.segments:
                seg_record = SegmentRecord(
                    transcription_id=tx_record.id,
                    segment_index=seg.id,
                    text=seg.text,
                    start_time=seg.start,
                    end_time=seg.end,
                    speaker=seg.speaker,
                )
                session.add(seg_record)

            session.commit()
            logger.info(
                "Saved live session #%d (%d segments)", tx_record.id, len(transcript.segments)
            )
            return tx_record.id

    def find_by_hash(self, file_hash: str) -> TranscriptionRecord | None:
        """Find an existing transcription by audio file hash (deduplication).

        Returns the most recent transcription for the given file hash, or None.
        """
        with get_session() as session:
            audio = session.query(AudioFileRecord).filter_by(file_hash=file_hash).first()
            if audio is None:
                return None

            return (
                session.query(TranscriptionRecord)
                .filter_by(audio_file_id=audio.id)
                .order_by(desc(TranscriptionRecord.created_at))
                .first()
            )

    def get_history(self, limit: int = 20, offset: int = 0, chapter_mode: bool | None = None) -> list[dict]:
        """Get recent transcription history.

        Args:
            chapter_mode: If True, returns only chapter transcripts. If False, returns only master transcripts. If None, returns all.
        Returns:
            List of dicts with transcription + audio metadata.
        """
        with get_session() as session:
            query = session.query(TranscriptionRecord)
            if chapter_mode is True:
                query = query.filter(TranscriptionRecord.chapter_id.isnot(None))
            elif chapter_mode is False:
                query = query.filter(TranscriptionRecord.chapter_id.is_(None))

            records = (
                query.order_by(desc(TranscriptionRecord.created_at))
                .offset(offset)
                .limit(limit)
                .all()
            )

            results = []
            for rec in records:
                if rec.file_name:
                    label = rec.file_name
                elif rec.source == "live":
                    label = "🎤 Live session"
                else:
                    audio = rec.audio_file
                    label = audio.file_name if audio else "unknown"
                results.append(
                    {
                        "id": rec.id,
                        "file_name": label,
                        "source": rec.source,
                        "language": rec.language,
                        "model": rec.model_name,
                        "word_count": rec.word_count,
                        "duration": rec.duration_seconds,
                        "status": rec.status,
                        "audio_file_id": rec.audio_file_id,
                        "chapter_id": getattr(rec, "chapter_id", None),
                        "refined_at": rec.refined_at.isoformat() if rec.refined_at else None,
                        "created_at": rec.created_at.isoformat() if rec.created_at else "",
                        "text_preview": rec.full_text[:100] + "..."
                        if len(rec.full_text) > 100
                        else rec.full_text,
                    }
                )

            return results

    def get_file_by_path(self, file_path: str) -> dict | None:
        """Find an audio file record by its absolute path."""
        from pathlib import Path
        abs_path = str(Path(file_path).absolute())
        with get_session() as session:
            rec = session.query(AudioFileRecord).filter_by(file_path=abs_path).first()
            if not rec:
                return None
            return {
                "id": rec.id,
                "file_path": rec.file_path,
                "file_name": rec.file_name,
                "duration_seconds": rec.duration_seconds,
                "format": rec.format,
                "file_size_bytes": rec.file_size_bytes,
                "created_at": rec.created_at.isoformat() if rec.created_at else "",
            }

    def get_audio_file(self, audio_file_id: int) -> dict | None:
        """Find an audio file record by its DB ID."""
        with get_session() as session:
            rec = session.query(AudioFileRecord).filter_by(id=audio_file_id).first()
            if not rec:
                return None
            
            # Count transcriptions
            tx_count = session.query(TranscriptionRecord).filter_by(audio_file_id=audio_file_id).count()
            
            return {
                "id": rec.id,
                "file_path": rec.file_path,
                "file_name": rec.file_name,
                "duration_seconds": rec.duration_seconds,
                "format": rec.format,
                "file_size_bytes": rec.file_size_bytes,
                "created_at": rec.created_at.isoformat() if rec.created_at else "",
                "transcript_count": tx_count,
            }

    def get_or_create_file(self, file_path: str) -> int:
        """Find an audio file record by path, or create a stub if it doesn't exist."""
        from pathlib import Path
        import os
        
        # Import to library first
        abs_path = self._import_to_library(file_path)
        
        with get_session() as session:
            rec = session.query(AudioFileRecord).filter_by(file_path=abs_path).first()
            if rec:
                return rec.id
                
            # Create a stub record. It will be filled in properly when transcribed.
            path_obj = Path(abs_path)
            file_size = os.path.getsize(abs_path) if path_obj.exists() else 0
            
            new_rec = AudioFileRecord(
                file_path=abs_path,
                file_name=path_obj.name,
                file_size_bytes=file_size,
                format=path_obj.suffix.lstrip("."),
            )
            session.add(new_rec)
            session.commit()
            return new_rec.id

    def get_latest_transcript_for_file(self, audio_file_id: int) -> dict | None:
        """Find the most recent completed transcription for a file."""
        with get_session() as session:
            rec = (
                session.query(TranscriptionRecord)
                .filter_by(audio_file_id=audio_file_id, status="completed")
                .order_by(desc(TranscriptionRecord.created_at))
                .first()
            )
            if not rec:
                return None
            return self.get_by_id(rec.id)

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Search transcriptions by text content.

        Args:
            query: Search string (case-insensitive LIKE).
            limit: Maximum number of results.

        Returns:
            List of matching transcription dicts.
        """
        with get_session() as session:
            pattern = f"%{query}%"
            records = (
                session.query(TranscriptionRecord)
                .filter(TranscriptionRecord.full_text.ilike(pattern))
                .order_by(desc(TranscriptionRecord.created_at))
                .limit(limit)
                .all()
            )

            return [
                {
                    "id": rec.id,
                    "file_name": rec.file_name
                    or (rec.audio_file.file_name if rec.audio_file else "unknown"),
                    "language": rec.language,
                    "text_preview": rec.full_text[:200],
                    "created_at": rec.created_at.isoformat() if rec.created_at else "",
                }
                for rec in records
            ]

    def get_by_id(self, transcription_id: int) -> dict | None:
        """Get full transcription by ID including all segments."""
        with get_session() as session:
            rec = session.query(TranscriptionRecord).filter_by(id=transcription_id).first()
            if rec is None:
                return None

            data = {
                "id": rec.id,
                "file_name": rec.file_name
                or (
                    "🎤 Live session"
                    if rec.source == "live"
                    else (rec.audio_file.file_name if rec.audio_file else "unknown")
                ),
                "file_path": (rec.audio_file.file_path if rec.audio_file else None),
                "audio_file_id": rec.audio_file_id,
                "source": rec.source,
                "full_text": rec.full_text,
                "raw_text": rec.raw_text or "",
                "language": rec.language,
                "language_probability": rec.language_probability,
                "engine": rec.engine,
                "model": rec.model_name,
                "duration": rec.duration_seconds,
                "word_count": rec.word_count,
                "segment_count": rec.segment_count,
                "status": rec.status,
                "refined_at": rec.refined_at.isoformat() if rec.refined_at else None,
                "created_at": rec.created_at.isoformat() if rec.created_at else "",
                "segments": [
                    {
                        "index": seg.segment_index,
                        "text": seg.text,
                        "start": seg.start_time,
                        "end": seg.end_time,
                        "speaker": seg.speaker,
                        "chapter_id": seg.chapter_id,
                    }
                    for seg in sorted(rec.segments, key=lambda s: s.segment_index)
                ],
                "speaker_map": json.loads(rec.speaker_map) if rec.speaker_map else {},
                "chapters": [],
            }

            # Fetch chapters if we have an audio file ID
            if rec.audio_file_id:
                from audiobench.storage.chapter_repository import get_chapter_repo
                chapter_records = get_chapter_repo().get_chapters(rec.audio_file_id)
                data["chapters"] = [c.to_dict() for c in chapter_records]

            return data

    def update_text(self, transcription_id: int, new_text: str) -> bool:
        """Update the full text of a transcription (used by REPL .edit).

        Returns True if found and updated, False if not found.
        """
        with get_session() as session:
            rec = session.query(TranscriptionRecord).filter_by(id=transcription_id).first()
            if rec is None:
                return False
            rec.full_text = new_text
            rec.word_count = len(new_text.split())
            session.commit()
            logger.info("Updated text for transcription #%d", transcription_id)
            return True

    def update_full_text(
        self,
        transcription_id: int,
        refined_text: str,
        raw_text: str,
    ) -> bool:
        """Update transcript with LLM-refined text, preserving the raw version.

        If raw_text is currently empty (pre-existing transcription), seeds it
        from the current full_text before overwriting. Stamps refined_at.

        Args:
            transcription_id: The transcription to update.
            refined_text: LLM-cleaned transcript text.
            raw_text: Original Whisper output (preserved in raw_text column).

        Returns:
            True if found and updated, False if not found.
        """
        with get_session() as session:
            rec = session.query(TranscriptionRecord).filter_by(id=transcription_id).first()
            if rec is None:
                return False
            # Seed raw_text for pre-existing transcriptions that weren't yet cleaned
            if not rec.raw_text:
                rec.raw_text = raw_text
            rec.full_text = refined_text
            rec.word_count = len(refined_text.split())
            rec.refined_at = datetime.now(UTC)
            session.commit()
            logger.info(
                "Refined transcript #%d (%d → %d chars)",
                transcription_id,
                len(raw_text),
                len(refined_text),
            )
            return True

    def update_segments(
        self,
        transcription_id: int,
        cleaned_texts: list[str],
    ) -> bool:
        """Bulk-update segment texts for a transcription (timestamps unchanged).

        Args:
            transcription_id: The transcription whose segments to update.
            cleaned_texts: New text for each segment, in segment_index order.

        Returns:
            True if all segments were updated, False if count mismatch or not found.
        """
        with get_session() as session:
            segments = (
                session.query(SegmentRecord)
                .filter_by(transcription_id=transcription_id)
                .order_by(SegmentRecord.segment_index)
                .all()
            )
            if not segments:
                return False
            if len(segments) != len(cleaned_texts):
                logger.warning(
                    "update_segments: segment count mismatch for #%d (%d vs %d)",
                    transcription_id,
                    len(segments),
                    len(cleaned_texts),
                )
                return False
            for seg, new_text in zip(segments, cleaned_texts, strict=True):
                seg.text = new_text
            session.commit()
            logger.info(
                "Updated %d segments for transcription #%d",
                len(segments),
                transcription_id,
            )
            return True

    def get_refinement_status(self, transcription_id: int) -> dict | None:
        """Return refinement status for a transcription.

        Returns:
            Dict with keys: is_refined, refined_at, raw_text, full_text.
            None if not found.
        """
        with get_session() as session:
            rec = session.query(TranscriptionRecord).filter_by(id=transcription_id).first()
            if rec is None:
                return None
            return {
                "id": rec.id,
                "is_refined": rec.refined_at is not None,
                "refined_at": rec.refined_at.isoformat() if rec.refined_at else None,
                "raw_text": rec.raw_text or "",
                "full_text": rec.full_text or "",
            }

    def delete_by_id(self, transcription_id: int) -> bool:
        """Delete a transcription by ID.

        Returns True if found and deleted, False if not found.
        """
        with get_session() as session:
            rec = session.query(TranscriptionRecord).filter_by(id=transcription_id).first()
            if rec is None:
                return False
            session.delete(rec)
            session.commit()
            logger.info("Deleted transcription #%d", transcription_id)
            return True

    def delete_all(self) -> int:
        """Delete all transcriptions. Returns number deleted."""
        with get_session() as session:
            count = session.query(TranscriptionRecord).count()
            session.query(SegmentRecord).delete()
            session.query(TranscriptionRecord).delete()
            session.commit()
            logger.info("Deleted %d transcription(s)", count)
            return count
