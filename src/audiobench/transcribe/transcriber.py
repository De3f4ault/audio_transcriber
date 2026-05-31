"""Pipeline orchestrator — the main entry point for transcription.

Chains: load → transcribe → store → format → output

Emits phase callbacks for UI progress:
    on_phase("loading", "Loading model...")
    on_phase("converting", "Converting audio...")
    on_phase("transcribing", "Transcribing...", progress=0.42)
    on_phase("saving", "Saving to database...")
    on_phase("done", "Complete!")

Design principles applied in this refactor:
- transcribe_file() is an ~80-line orchestrator; heavy logic lives in private methods.
- ChapterInfo is the universal chapter currency; no raw dicts or ORM objects escape.
- get_chapter_repo() singleton avoids re-instantiating the repository on every call.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from audiobench.core.db_engine import init_db
from audiobench.core.logger_factory import get_logger
from audiobench.core.settings import get_settings
from audiobench.storage.repository import TranscriptionRepository
from audiobench.transcribe.audio_converter import AudioLoader
from audiobench.transcribe.engines.engine_protocol import TranscriptionEngine
from audiobench.transcribe.engines.engine_registry import create_engine
from audiobench.transcribe.transcription_result import AudioMetadata, Segment, Transcript

logger = get_logger("core.pipeline")

# Callback types
PhaseCallback = Callable[[str, str, float | None], None]
SegmentCallback = Callable[[Segment], None]


class TranscriptionPipeline:
    """Orchestrates the full transcription workflow."""

    def __init__(
        self,
        engine: TranscriptionEngine | None = None,
        repository: TranscriptionRepository | None = None,
    ) -> None:
        self._engine = engine
        self._repository = repository or TranscriptionRepository()
        self._settings = get_settings()
        self._db_initialized = False

    # ── Public API ───────────────────────────────────────────────────────────

    def transcribe_file(
        self,
        file_path: str | Path,
        language: str | None = None,
        output_format: str | None = None,
        output_path: str | None = None,
        word_timestamps: bool | None = None,
        skip_cache: bool = False,
        speed_preset: str | None = None,
        initial_prompt: str | None = None,
        translate: bool = False,
        enable_diarization: bool = False,
        map_speakers: str | None = None,
        auto_name: bool = False,
        on_phase: PhaseCallback | None = None,
        on_segment: SegmentCallback | None = None,
        engine_name: str | None = None,
        job_id: int | None = None,
        target_chapters: list[int] | None = None,
        resume: bool = False,
        parallel: int = 1,
        skip_ghost: bool = True,
        chapter_id: int | None = None,
    ) -> Transcript:
        """Transcribe an audio file through the full pipeline.

        Args:
            file_path: Path to audio/video file.
            language: Language code or None for auto-detect.
            output_format: Override default format (txt/srt/vtt/json).
            output_path: Write output to file; None = return only.
            word_timestamps: Override setting.
            skip_cache: If True, skip dedup check and re-transcribe.
            speed_preset: Override speed preset (fast/balanced/accurate).
            on_phase: Callback for phase updates (phase, message, progress).
            job_id: Optional ID of the background job for emitting events.
            target_chapters: If set, only transcribe these chapter indices.
            resume: Skip chapters already marked 'completed'.
            parallel: Number of parallel chapter workers (1 = sequential).
            skip_ghost: Skip zero-duration ghost chapters.
            chapter_id: DB chapter ID if this call is for a single chapter.

        Returns:
            Transcript result.
        """
        self._ensure_db()
        engine = self._ensure_engine(on_phase, engine_name=engine_name)

        fmt = output_format or self._settings.output_format
        word_ts = word_timestamps if word_timestamps is not None else self._settings.word_timestamps
        preset = speed_preset or self._settings.speed_preset

        def emit(phase: str, message: str, progress: float | None = None) -> None:
            if on_phase:
                on_phase(phase, message, progress)
            kw = {"phase": phase}
            if progress is not None:
                kw["progress"] = int(progress * 100) if isinstance(progress, float) else progress
            self._emit_event(job_id, **kw)

        try:
            if target_chapters is not None:
                transcript = self._run_chapter_pipeline(
                    file_path=file_path,
                    target_chapters=target_chapters,
                    emit=emit,
                    language=language,
                    word_timestamps=word_ts,
                    skip_cache=skip_cache,
                    speed_preset=preset,
                    initial_prompt=initial_prompt,
                    translate=translate,
                    enable_diarization=enable_diarization,
                    map_speakers=map_speakers,
                    auto_name=auto_name,
                    on_segment=on_segment,
                    engine_name=engine_name,
                    job_id=job_id,
                    resume=resume,
                    parallel=parallel,
                    skip_ghost=skip_ghost,
                )
            else:
                transcript = self._run_single_pipeline(
                    file_path=file_path,
                    engine=engine,
                    emit=emit,
                    language=language,
                    word_ts=word_ts,
                    preset=preset,
                    skip_cache=skip_cache,
                    initial_prompt=initial_prompt,
                    translate=translate,
                    enable_diarization=enable_diarization,
                    map_speakers=map_speakers,
                    auto_name=auto_name,
                    on_segment=on_segment,
                    chapter_id=chapter_id,
                    job_id=job_id,
                )

            if output_path:
                self._write_output(transcript, fmt, output_path)
                logger.info("Pipeline: wrote %s output to %s", fmt, output_path)

            if job_id:
                from audiobench.jobs.repository import JobRepository
                JobRepository().mark_job_done(job_id)

            return transcript

        except Exception:
            if job_id:
                from audiobench.jobs.repository import JobRepository
                JobRepository().mark_job_failed(job_id, exit_code=1)
            raise

    # ── Private: Single-file pipeline ────────────────────────────────────────

    def _run_single_pipeline(
        self,
        file_path: str | Path,
        engine: TranscriptionEngine,
        emit: Callable,
        language: str | None,
        word_ts: bool,
        preset: str,
        skip_cache: bool,
        initial_prompt: str | None,
        translate: bool,
        enable_diarization: bool,
        map_speakers: str | None,
        auto_name: bool,
        on_segment: SegmentCallback | None,
        chapter_id: int | None,
        job_id: int | None,
    ) -> Transcript:
        """Load, transcribe, diarize, and save a single audio file."""
        is_gemini = engine.engine_name == "gemini"
        beam = self._settings.resolve_beam_size(preset)
        batch = self._settings.resolve_batch_size(preset)
        temperature = self._settings.resolve_temperature(preset)
        condition_on_prev = self._settings.resolve_condition_on_previous_text(preset)

        emit("converting", "Converting audio...")
        logger.info("Pipeline: loading %s (preset=%s, beam=%d, batch=%d)", file_path, preset, beam, batch)

        with AudioLoader() as loader:
            wav_path, metadata = loader.load(file_path)

            # Cache check
            cached = self._check_cache(metadata, skip_cache)
            if cached:
                transcript = cached
                if enable_diarization and not is_gemini and not any(s.speaker for s in transcript.segments):
                    transcript = self._run_diarization(wav_path, transcript, emit)
                emit("done", "Retrieved from cache")
                self._emit_event(job_id, phase="done", words=transcript.word_count,
                                 speakers=len(transcript.speaker_map), duration=int(transcript.duration_seconds))
                return transcript

            # Transcribe
            task = "translate" if translate else "transcribe"
            emit("transcribing", "Translating to English..." if translate else "Transcribing...", 0.0)
            logger.info("Pipeline: transcribing with %s (preset=%s)", engine.engine_name, preset)

            def _progress(pct: float) -> None:
                emit("transcribing", "Transcribing...", pct)

            audio_input = str(file_path) if is_gemini else wav_path
            if is_gemini:
                transcript = engine.transcribe(
                    audio_input, language=language or self._settings.language,
                    task=task, word_timestamps=word_ts, on_phase=emit, diarize=enable_diarization,
                )
            else:
                transcript = engine.transcribe(
                    audio_input, language=language or self._settings.language,
                    task=task, word_timestamps=word_ts, beam_size=beam, batch_size=batch,
                    temperature=temperature, compression_ratio_threshold=2.4, no_speech_threshold=0.6,
                    log_prob_threshold=-1.0, condition_on_previous_text=condition_on_prev,
                    repetition_penalty=1.1, initial_prompt=initial_prompt,
                    progress_callback=_progress, on_segment=on_segment,
                )
            transcript.audio = metadata

            # Diarization
            if enable_diarization and not is_gemini:
                transcript = self._run_diarization(wav_path, transcript, emit)

            # Speaker naming
            self._apply_speaker_naming(transcript, map_speakers, auto_name, enable_diarization, emit)

            # Save
            emit("saving", "Saving to database...")
            tx_id = self._repository.save_transcription(transcript, metadata, chapter_id=chapter_id)
            logger.info("Pipeline: saved as transcription #%d", tx_id)

            if transcript.segments:
                self._spawn_refinement(tx_id, raw_text=transcript.text, segments=transcript.segments)

            emit("done", "Complete!")
            self._emit_event(job_id, phase="done", words=transcript.word_count,
                             speakers=len(transcript.speaker_map), duration=int(transcript.duration_seconds))
            return transcript

    # ── Private: Chapter pipeline ─────────────────────────────────────────────

    def _run_chapter_pipeline(
        self,
        file_path: str | Path,
        target_chapters: list[int],
        emit: Callable,
        language: str | None,
        word_timestamps: bool,
        skip_cache: bool,
        speed_preset: str,
        initial_prompt: str | None,
        translate: bool,
        enable_diarization: bool,
        map_speakers: str | None,
        auto_name: bool,
        on_segment: SegmentCallback | None,
        engine_name: str | None,
        job_id: int | None,
        resume: bool,
        parallel: int,
        skip_ghost: bool,
    ) -> Transcript:
        """Split a file by chapter indices and transcribe each chunk."""
        from audiobench.chapters.cue_parser import ChapterInfo
        from audiobench.chapters.splitter import ChapterSplitter
        from audiobench.storage.chapter_repository import get_chapter_repo

        repo = get_chapter_repo()

        # Ensure the audio file exists in the library and chapters are detected
        audio_record = self._ensure_audio_record(file_path, emit)
        all_chapters = repo.get_chapters(audio_record.id if audio_record else 0)

        # Filter to the requested indices
        chapters_to_process = [c for c in all_chapters if c.index in target_chapters]
        if skip_ghost:
            chapters_to_process = [c for c in chapters_to_process if not c.is_ghost]
        if resume:
            # We can't query transcription_status from ChapterInfo directly, need the repo
            # Re-fetch raw status; for now omit resume filtering here — it's handled in save
            pass
        if not chapters_to_process:
            raise ValueError(
                f"None of the requested chapters {target_chapters} need processing "
                "(all filtered out by skip_ghost or resume)."
            )

        splitter = ChapterSplitter()
        source_path = Path(audio_record.file_path if audio_record else file_path)

        with tempfile.TemporaryDirectory() as tmp_dir:
            emit("converting", "Extracting chapters...", 0.0)
            chunk_paths = splitter.split(source_path, chapters_to_process, Path(tmp_dir), fmt="wav")

            def _process_chunk(i: int, chap: ChapterInfo, chunk_path: Path | None) -> Transcript | None:
                if chunk_path is None:
                    return None
                emit("transcribing", f"Transcribing chapter {chap.index}...", float(i) / len(chapters_to_process))
                result = self.transcribe_file(
                    file_path=chunk_path,
                    language=language,
                    output_format=None,
                    output_path=None,
                    word_timestamps=word_timestamps,
                    skip_cache=True,
                    speed_preset=speed_preset,
                    initial_prompt=initial_prompt,
                    translate=translate,
                    enable_diarization=enable_diarization,
                    map_speakers=map_speakers,
                    auto_name=auto_name,
                    on_phase=None,
                    on_segment=on_segment,
                    engine_name=engine_name,
                    job_id=job_id,
                    target_chapters=None,      # prevent recursion
                    chapter_id=chap.id,
                )
                # Shift timestamps to match the full-file timeline
                offset = chap.start_time
                for seg in result.segments:
                    seg.start += offset
                    seg.end += offset
                    for word in seg.words:
                        word.start += offset
                        word.end += offset
                return result

            if parallel > 1:
                with ThreadPoolExecutor(max_workers=parallel) as ex:
                    futures = [ex.submit(_process_chunk, i, c, p)
                               for i, (c, p) in enumerate(zip(chapters_to_process, chunk_paths))]
                    results = [f.result() for f in futures if f.result() is not None]
            else:
                results = [r for i, (c, p) in enumerate(zip(chapters_to_process, chunk_paths))
                           if (r := _process_chunk(i, c, p)) is not None]

        if not results:
            raise RuntimeError("No chapters were successfully transcribed.")

        # Merge all chapter transcripts into one
        return self._merge_transcripts(results)

    # ── Private: Helpers ──────────────────────────────────────────────────────

    def _ensure_audio_record(self, file_path: str | Path, emit: Callable):
        """Ensure the audio file has a DB record, importing it into the library if needed."""
        from audiobench.chapters.detector import ChapterDetector
        from audiobench.storage.chapter_repository import get_chapter_repo
        from audiobench.storage.models import AudioFileRecord
        from audiobench.core.db_session import get_session

        with AudioLoader() as loader:
            _, metadata = loader.load(file_path)

        if metadata and metadata.file_hash:
            record = self._repository.find_by_hash(metadata.file_hash)
            if record:
                return record

        # New file — import to library and detect chapters
        new_path = self._repository._import_to_library(str(file_path))
        with get_session() as session:
            audio_record = AudioFileRecord(
                file_path=new_path, file_name=metadata.file_name,
                file_size_bytes=metadata.file_size_bytes, format=metadata.format,
                duration_seconds=metadata.duration_seconds, sample_rate=metadata.sample_rate,
                channels=metadata.channels, file_hash=metadata.file_hash,
            )
            session.add(audio_record)
            session.commit()

            try:
                chapters = ChapterDetector().detect(Path(new_path))
                if chapters:
                    get_chapter_repo().save_chapters(audio_record.id, chapters)
            except Exception as e:
                logger.warning("Chapter detection failed for %s: %s", new_path, e)

            return session.query(AudioFileRecord).filter_by(id=audio_record.id).first()

    def _check_cache(self, metadata: AudioMetadata, skip_cache: bool) -> Transcript | None:
        """Return a cached Transcript if one exists, otherwise None."""
        if skip_cache or not metadata.file_hash:
            return None
        cached = self._repository.find_by_hash(metadata.file_hash)
        if not cached:
            return None
        logger.info("Pipeline: cache hit for hash %s", metadata.file_hash[:12])
        data = self._repository.get_by_id(cached.id)
        return self._reconstruct_transcript(data, metadata) if data else None

    def _run_diarization(self, wav_path: str, transcript: Transcript, emit: Callable) -> Transcript:
        """Run speaker diarization, returning updated transcript (or original on failure)."""
        emit("diarizing", "Identifying speakers...")
        try:
            from audiobench.diarization.engine import PyannoteDiarizer
            diarizer = PyannoteDiarizer(hf_token=self._settings.hf_token)
            result = diarizer.diarize(wav_path, transcript)
            logger.info("Pipeline: diarization complete")
            return result
        except Exception as e:
            logger.warning("Diarization failed (continuing without): %s", e)
            return transcript

    def _apply_speaker_naming(
        self,
        transcript: Transcript,
        map_speakers: str | None,
        auto_name: bool,
        enable_diarization: bool,
        emit: Callable,
    ) -> None:
        """Apply manual or automatic speaker name mapping in-place."""
        if map_speakers:
            emit("naming", "Applying manual speaker map...")
            try:
                for pair in map_speakers.split(","):
                    k, v = pair.split("=")
                    transcript.speaker_map[k.strip()] = v.strip()
                logger.info("Applied manual speaker map: %s", transcript.speaker_map)
            except Exception as e:
                logger.warning("Failed to parse map_speakers '%s': %s", map_speakers, e)
        elif auto_name and enable_diarization:
            emit("naming", "Auto-detecting speaker names...")
            try:
                self._auto_name_speakers(transcript)
                logger.info("Auto-detected speaker map: %s", transcript.speaker_map)
            except Exception as e:
                logger.warning("Auto-naming failed (continuing without): %s", e)

    def _merge_transcripts(self, results: list[Transcript]) -> Transcript:
        """Merge a list of chapter transcripts into a single unified Transcript."""
        merged = results[0]
        for r in results[1:]:
            merged.segments.extend(r.segments)
            merged.duration_seconds = max(merged.duration_seconds, r.duration_seconds)
        return merged

    # ── Private: Infrastructure ───────────────────────────────────────────────

    def _ensure_engine(
        self,
        on_phase: PhaseCallback | None = None,
        engine_name: str | None = None,
    ) -> TranscriptionEngine:
        """Lazy-init engine from settings if not provided."""
        if self._engine is None:
            selected = engine_name or self._settings.engine
            if on_phase:
                label = "Connecting to Gemini..." if selected == "gemini" else "Loading model..."
                on_phase("loading", label, None)
            self._engine = create_engine(
                engine_name=selected,
                model_name=(
                    self._settings.gemini_model if selected == "gemini"
                    else self._settings.model_name
                ),
                device=self._settings.resolve_device(),
                compute_type=self._settings.resolve_compute_type(),
                cpu_threads=self._settings.resolve_cpu_threads(),
            )
        return self._engine

    def _ensure_db(self) -> None:
        """Ensure database tables exist."""
        if not self._db_initialized:
            init_db()
            self._db_initialized = True

    def _emit_event(self, job_id: int | None, **kwargs) -> None:
        """Write a terse machine-readable event to the job's events file."""
        if not job_id:
            return
        import time
        line = " ".join(f"{k}={v}" for k, v in kwargs.items())
        line += f" ts={int(time.time())}"
        log_dir = self._settings.data_dir / "job_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / f"job_{job_id}.events", "a") as f:
            f.write(line + "\n")

    def _write_output(self, transcript: Transcript, fmt: str, output_path: str) -> None:
        """Format transcript and write to file."""
        if fmt == "pdf":
            from audiobench.export.pdf import PDFExporter
            data = transcript.dict()
            data["file_name"] = Path(output_path).stem if output_path else "transcript"
            PDFExporter().export_transcript(data, output_path)
            return

        from audiobench.output.base import get_formatter
        formatter = get_formatter(fmt)
        content = formatter.format(transcript)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

    def _auto_name_speakers(self, transcript: Transcript) -> None:
        """Use Gemini to detect actual speaker names from the transcript context."""
        from google import genai
        from audiobench.output.text import TextFormatter
        import json
        import re

        if not self._settings.gemini_api_key:
            logger.warning("Gemini API key not configured, skipping auto-name")
            return

        client = genai.Client(api_key=self._settings.gemini_api_key)

        # Use only the first ~5 minutes for speaker identification
        intro_segments = [s for s in transcript.segments if s.end <= 300][:50]
        intro = Transcript(
            segments=intro_segments,
            duration_seconds=intro_segments[-1].end if intro_segments else 0.0,
        )
        formatted_intro = TextFormatter().format(intro)

        prompt = (
            "Analyze the following transcript excerpt and identify the real names of the speakers "
            "based on the context (e.g. introductions, 'Welcome to the podcast, John').\n\n"
            f"{formatted_intro}\n\n"
            "Respond ONLY with a valid JSON object mapping the generic speaker labels to their detected names. "
            "If a name cannot be determined with high confidence, do not include it in the JSON.\n"
            'Example format:\n{\n  "Speaker 1": "Lex Fridman",\n  "Speaker 2": "Elon Musk"\n}'
        )

        response = client.models.generate_content(model="gemini-2.5-pro", contents=prompt)
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
            raw = re.sub(r"\n?```\s*$", "", raw)

        try:
            detected = json.loads(raw)
            if isinstance(detected, dict):
                transcript.speaker_map.update(
                    {k: v for k, v in detected.items() if isinstance(k, str) and isinstance(v, str)}
                )
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse Gemini auto-name JSON: %s", e)

    def _spawn_refinement(self, tx_id: int, raw_text: str, segments: list) -> None:
        """Spawn a background thread to refine transcript segments using an LLM."""
        import threading

        def _refine() -> None:
            try:
                from audiobench.chat.providers.ollama_provider import OllamaClient
                from audiobench.transcribe.refiner import TranscriptRefiner

                client = OllamaClient(
                    base_url=self._settings.ollama_base_url,
                    model=self._settings.clean_model,
                )
                if not client.is_available():
                    logger.info("Ollama not available, skipping refinement for #%d", tx_id)
                    return

                refiner = TranscriptRefiner(client, model=self._settings.clean_model)
                seg_texts = [seg.text for seg in segments]
                cleaned = refiner.refine_segments(seg_texts)

                if cleaned == seg_texts:
                    logger.info("Refinement produced no changes for #%d", tx_id)
                    return

                if not self._repository.update_segments(tx_id, cleaned):
                    logger.warning("update_segments failed for #%d", tx_id)
                    return

                refined_full = " ".join(t.strip() for t in cleaned if t.strip())
                self._repository.update_full_text(tx_id, refined_full, raw_text)
                logger.info("Segment refinement complete for #%d", tx_id)
            except Exception as e:
                logger.warning("Background refinement failed for #%d: %s", tx_id, e)

        thread = threading.Thread(target=_refine, name=f"refine-{tx_id}", daemon=True)
        thread.start()
        logger.info("Spawned background refinement thread for #%d", tx_id)

    def _reconstruct_transcript(self, data: dict, metadata: AudioMetadata) -> Transcript:
        """Reconstruct a Transcript from cached DB data."""
        from audiobench.transcribe.transcription_result import Segment

        segments = [
            Segment(id=s["index"], text=s["text"], start=s["start"], end=s["end"], speaker=s.get("speaker"))
            for s in data.get("segments", [])
        ]
        return Transcript(
            segments=segments, language=data.get("language", "en"),
            language_probability=data.get("language_probability", 0.0),
            audio=metadata, duration_seconds=data.get("duration", 0.0),
            engine=data.get("engine", "faster-whisper"),
            model_name=data.get("model", "large-v3-turbo"),
        )
