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
from audiobench.transcribe.checkpoint_manager import CheckpointManager

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
        filters: list[str] | None = None,
        engine_name: str | None = None,
        job_id: int | None = None,
        target_chapters: list[int] | None = None,
        resume: bool = False,
        strategy: str = "batch",
        pipeline_workers: int = 2,
        parallel: int = 1,
        skip_ghost: bool = True,
        chapter_id: int | None = None,
        diarize_mode: str = "fast",
        diarize_threshold: float = 0.65,
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
                    filters=filters,
                    engine_name=engine_name,
                    job_id=job_id,
                    resume=resume,
                    strategy=strategy,
                    pipeline_workers=pipeline_workers,
                    parallel=parallel,
                    skip_ghost=skip_ghost,
                    diarize_mode=diarize_mode,
                    diarize_threshold=diarize_threshold,
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
                    filters=filters,
                    chapter_id=chapter_id,
                    job_id=job_id,
                    diarize_mode=diarize_mode,
                    diarize_threshold=diarize_threshold,
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
        filters: list[str] | None,
        chapter_id: int | None,
        job_id: int | None,
        diarize_mode: str = "fast",
        diarize_threshold: float = 0.65,
    ) -> Transcript:
        """Load, transcribe, diarize, and save a single audio file."""
        is_gemini = engine.engine_name == "gemini"
        beam = self._settings.resolve_beam_size(preset)
        batch = self._settings.resolve_batch_size(preset)
        temperature = self._settings.resolve_temperature(preset)
        condition_on_prev = self._settings.resolve_condition_on_previous_text(preset)

        emit("converting", "Converting audio...")
        logger.info(
            "Pipeline: loading %s (preset=%s, beam=%d, batch=%d)", file_path, preset, beam, batch
        )

        with AudioLoader() as loader:
            wav_path, metadata = loader.load(file_path, filters=filters)

            # Cache check
            cached = self._check_cache(metadata, skip_cache)
            if cached:
                transcript = cached
                if (
                    enable_diarization
                    and not is_gemini
                    and not any(s.speaker for s in transcript.segments)
                ):
                    transcript = self._run_diarization(wav_path, transcript, emit, diarize_mode, diarize_threshold)
                emit("done", "Retrieved from cache")
                self._emit_event(
                    job_id,
                    phase="done",
                    words=transcript.word_count,
                    speakers=len(transcript.speaker_map),
                    duration=int(transcript.duration_seconds),
                )
                return transcript

            # Transcribe
            task = "translate" if translate else "transcribe"
            emit("transcribing", "Translating to English..." if translate else "Transcribing...", 0.0)
            logger.info("Pipeline: transcribing with %s (preset=%s)", engine.engine_name, preset)

            def _progress(pct: float) -> None:
                emit("transcribing", "Transcribing...", pct)

            def _do_transcribe():
                if is_gemini:
                    return engine.transcribe(
                        wav_path,
                        language=language or self._settings.language,
                        task=task,
                        word_timestamps=word_ts,
                        on_phase=emit,
                        diarize=enable_diarization,
                    )
                else:
                    return engine.transcribe(
                        wav_path,
                        language=language or self._settings.language,
                        task=task,
                        word_timestamps=word_ts,
                        beam_size=beam,
                        batch_size=batch,
                        temperature=temperature,
                        compression_ratio_threshold=2.4,
                        no_speech_threshold=0.6,
                        log_prob_threshold=-1.0,
                        condition_on_previous_text=condition_on_prev,
                        repetition_penalty=1.1,
                        initial_prompt=initial_prompt,
                        progress_callback=_progress,
                        on_segment=on_segment,
                    )

            diarize_device = self._settings.resolve_diarization_device()
            whisper_device_index = self._settings.resolve_device_index()
            whisper_dev = f"cuda:{whisper_device_index}" if isinstance(whisper_device_index, int) else f"cuda:{whisper_device_index[0]}"
            
            is_concurrent_capable = (
                enable_diarization 
                and not is_gemini
                and diarize_mode == "accurate"
                and diarize_device.startswith("cuda")
                and (whisper_dev != diarize_device)
            )

            if is_concurrent_capable:
                logger.info("Running transcription and diarization concurrently on separate GPUs")
                from audiobench.diarization.engine import PyannoteDiarizer
                diarizer = PyannoteDiarizer(hf_token=self._settings.hf_token, device=diarize_device)
                
                with ThreadPoolExecutor(max_workers=2) as ex:
                    ft = ex.submit(_do_transcribe)
                    fd = ex.submit(diarizer.get_speaker_turns, wav_path)
                    
                    transcript = ft.result()
                    transcript.audio = metadata
                    
                    try:
                        turns = fd.result()
                        transcript = diarizer.assign_speakers(transcript, turns, audio_path=wav_path)
                        logger.info("Pipeline: diarization complete")
                    except Exception as e:
                        logger.warning("Concurrent diarization failed (continuing without): %s", e)
            else:
                transcript = _do_transcribe()
                transcript.audio = metadata
                if enable_diarization and not is_gemini:
                    transcript = self._run_diarization(wav_path, transcript, emit, diarize_mode, diarize_threshold)

            # Speaker naming
            self._apply_speaker_naming(
                transcript, map_speakers, auto_name, enable_diarization, emit
            )

            # Save
            emit("saving", "Saving to database...")
            tx_id = self._repository.save_transcription(
                transcript, 
                metadata, 
                chapter_id=chapter_id, 
                on_phase=emit, 
                overwrite=skip_cache
            )
            logger.info("Pipeline: saved as transcription #%d", tx_id)

            # Fire plugin hook
            try:
                from audiobench.events import get_bus
                get_bus().emit(
                    "transcription.complete",
                    tx_id=tx_id,
                    file_path=str(file_path),
                    duration_seconds=transcript.duration_seconds,
                    word_count=transcript.word_count,
                    language=transcript.language,
                )
            except Exception:
                logger.warning("EventBus emit failed (non-fatal)", exc_info=True)

            if transcript.segments:
                self._spawn_refinement(
                    tx_id, raw_text=transcript.text, segments=transcript.segments
                )
                if chapter_id is None:
                    self._spawn_auto_naming(tx_id, str(file_path), transcript)

            emit("done", "Complete!")
            self._emit_event(
                job_id,
                phase="done",
                words=transcript.word_count,
                speakers=len(transcript.speaker_map),
                duration=int(transcript.duration_seconds),
            )
            return transcript

    # ── Private: Chapter pipeline ─────────────────────────────────────────────

    def _run_chapter_pipeline(
        self,
        file_path: str | Path,
        target_chapters: list[int] | str,
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
        filters: list[str] | None,
        engine_name: str | None,
        job_id: int | None,
        resume: bool,
        strategy: str,
        pipeline_workers: int,
        parallel: int,
        skip_ghost: bool,
        diarize_mode: str = "fast",
        diarize_threshold: float = 0.65,
    ) -> Transcript:
        """Split a file by chapter indices and transcribe each chunk."""
        from audiobench.chapters.cue_parser import ChapterInfo
        from audiobench.chapters.splitter import ChapterSplitter
        from audiobench.storage.chapter_repository import get_chapter_repo

        repo = get_chapter_repo()

        # Ensure the audio file exists in the library and chapters are detected
        audio_record = self._ensure_audio_record(file_path, emit)
        all_chapters = repo.get_chapters(audio_record.id if audio_record else 0)

        cm = CheckpointManager(file_path)

        # Filter to the requested indices
        if target_chapters == "all":
            chapters_to_process = all_chapters
        else:
            chapters_to_process = [c for c in all_chapters if c.index in target_chapters]
        
        if skip_ghost:
            chapters_to_process = [c for c in chapters_to_process if not c.is_ghost]
            
        if resume:
            chapters_to_process = [c for c in chapters_to_process if not cm.has_checkpoint(c.index)]
            if len(chapters_to_process) < len(all_chapters):
                logger.info("Resuming: Skipped %d already-completed chapters.", len(all_chapters) - len(chapters_to_process))

        if not chapters_to_process and target_chapters:
            # Everything is already done! Load all checkpoints.
            results = [cm.load_checkpoint(c.index) for c in all_chapters]
            results = [r for r in results if r is not None]
            if not results:
                raise RuntimeError("No checkpoints found, but resume filtered all chapters.")
            emit("done", "Loaded all from cache")
            return self._merge_transcripts(results)

        splitter = ChapterSplitter()
        source_path = Path(audio_record.file_path if audio_record else file_path)

        with tempfile.TemporaryDirectory() as tmp_dir:
            emit("converting", "Extracting chapters...", 0.0)
            chunk_paths = splitter.split(source_path, chapters_to_process, Path(tmp_dir), fmt="wav")

            def _process_chunk(
                i: int, chap: ChapterInfo, chunk_path: Path | None, do_diarize: bool
            ) -> Transcript | None:
                if chunk_path is None:
                    return None
                
                # Check checkpoint
                if resume and cm.has_checkpoint(chap.index):
                    res = cm.load_checkpoint(chap.index)
                    if res:
                        return res

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
                    enable_diarization=do_diarize,
                    map_speakers=map_speakers,
                    auto_name=auto_name,
                    on_phase=None,
                    on_segment=on_segment,
                    filters=filters,
                    engine_name=engine_name,
                    job_id=job_id,
                    target_chapters=None,
                    chapter_id=chap.id,
                    diarize_mode=diarize_mode,
                    diarize_threshold=diarize_threshold,
                )
                
                # Shift timestamps
                offset = chap.start_time
                for seg in result.segments:
                    seg.start += offset
                    seg.end += offset
                    for word in seg.words:
                        word.start += offset
                        word.end += offset
                
                # Save checkpoint
                cm.save_checkpoint(chap.index, result)
                return result

            results = []
            
            if strategy == "batch":
                # Phase 1: Transcribe all
                for i, (c, p) in enumerate(zip(chapters_to_process, chunk_paths)):
                    r = _process_chunk(i, c, p, do_diarize=False)
                    if r: results.append(r)
                
                # Phase 2: Diarize all
                if enable_diarization:
                    emit("diarizing", "Diarizing all chapters...", 0.0)
                    for i, (c, p) in enumerate(zip(chapters_to_process, chunk_paths)):
                        res = cm.load_checkpoint(c.index)
                        if res and p and p.exists() and not any(s.speaker for s in res.segments):
                            emit("diarizing", f"Diarizing chapter {c.index}...", float(i) / len(chapters_to_process))
                            # Diarize and overwrite checkpoint
                            res = self._run_diarization(p, res, emit, diarize_mode, diarize_threshold)
                            cm.save_checkpoint(c.index, res)
            
            elif strategy == "concurrent":
                # Producer-Consumer pipeline for multi-GPU
                import queue
                import threading
                import torch
                
                diarize_device = self._settings.resolve_diarization_device()
                whisper_device_index = self._settings.resolve_device_index()
                whisper_dev = f"cuda:{whisper_device_index}" if isinstance(whisper_device_index, int) else f"cuda:{whisper_device_index[0]}"
                
                is_concurrent = (
                    enable_diarization
                    and diarize_mode == "accurate"
                    and diarize_device.startswith("cuda")
                    and (whisper_dev != diarize_device)
                )

                if is_concurrent:
                    logger.info("Using true producer-consumer concurrent chapter pipeline")
                    q = queue.Queue()
                    results = [None] * len(chapters_to_process)
                    
                    def producer():
                        for i, (c, p) in enumerate(zip(chapters_to_process, chunk_paths)):
                            r = _process_chunk(i, c, p, do_diarize=False)
                            q.put((i, c, p, r))
                        q.put(None)  # Sentinel
                        
                    def consumer():
                        while True:
                            item = q.get()
                            if item is None:
                                break
                            i, c, p, r = item
                            if r is not None:
                                if enable_diarization and p and p.exists() and not any(s.speaker for s in r.segments):
                                    emit("diarizing", f"Diarizing chapter {c.index}...", float(i) / len(chapters_to_process))
                                    r = self._run_diarization(p, r, emit, diarize_mode, diarize_threshold)
                                    cm.save_checkpoint(c.index, r)
                                results[i] = r
                            q.task_done()
                            
                    t1 = threading.Thread(target=producer)
                    t2 = threading.Thread(target=consumer)
                    t1.start()
                    t2.start()
                    t1.join()
                    t2.join()
                    results = [r for r in results if r is not None]
                else:
                    # Single-GPU sequential fallback but using threads to share memory/IO efficiently
                    torch.set_num_threads(1)
                    
                    def _process_concurrent(idx: int, c: ChapterInfo, p: Path) -> Transcript | None:
                        # Whisper
                        r = _process_chunk(idx, c, p, do_diarize=False)
                        if not r: return None
                        
                        # Pyannote
                        if enable_diarization and p and p.exists() and not any(s.speaker for s in r.segments):
                            emit("diarizing", f"Diarizing chapter {c.index}...", float(idx) / len(chapters_to_process))
                            r = self._run_diarization(p, r, emit, diarize_mode, diarize_threshold)
                            cm.save_checkpoint(c.index, r)
                        return r
                    
                    with ThreadPoolExecutor(max_workers=pipeline_workers) as ex:
                        futures = [ex.submit(_process_concurrent, i, c, p) for i, (c, p) in enumerate(zip(chapters_to_process, chunk_paths))]
                        results = [f.result() for f in futures if f.result() is not None]
                    
            else:
                # strategy == "chunk"
                if parallel > 1:
                    with ThreadPoolExecutor(max_workers=parallel) as ex:
                        futures = [ex.submit(_process_chunk, i, c, p, enable_diarization) for i, (c, p) in enumerate(zip(chapters_to_process, chunk_paths))]
                        results = [f.result() for f in futures if f.result() is not None]
                else:
                    results = [r for i, (c, p) in enumerate(zip(chapters_to_process, chunk_paths)) if (r := _process_chunk(i, c, p, enable_diarization)) is not None]

        # Load all checkpoints (including those skipped via resume)
        final_results = []
        for c in all_chapters:
            r = cm.load_checkpoint(c.index)
            if r: final_results.append(r)

        if not final_results:
            raise RuntimeError("No chapters were successfully transcribed.")

        return self._merge_transcripts(final_results)

    # ── Private: Helpers ──────────────────────────────────────────────────────

    def _ensure_audio_record(self, file_path: str | Path, emit: Callable):
        """Ensure the audio file has a DB record, importing it into the library if needed."""
        from audiobench.chapters.detector import ChapterDetector
        from audiobench.core.db_session import get_session
        from audiobench.storage.chapter_repository import get_chapter_repo
        from audiobench.storage.models import AudioFileRecord

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
                file_path=new_path,
                file_name=metadata.file_name,
                file_size_bytes=metadata.file_size_bytes,
                format=metadata.format,
                duration_seconds=metadata.duration_seconds,
                sample_rate=metadata.sample_rate,
                channels=metadata.channels,
                file_hash=metadata.file_hash,
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

    def _run_diarization(self, wav_path: str, transcript: Transcript, emit: Callable, diarize_mode: str = "fast", diarize_threshold: float = 0.65) -> Transcript:
        """Run speaker diarization, returning updated transcript (or original on failure)."""
        emit("diarizing", "Identifying speakers...")
        try:
            if diarize_mode == "accurate":
                from audiobench.diarization.engine import PyannoteDiarizer
                diarizer = PyannoteDiarizer(hf_token=self._settings.hf_token, device=self._settings.resolve_diarization_device())
            else:
                from audiobench.diarization.engine import LightweightDiarizer
                diarizer = LightweightDiarizer(distance_threshold=diarize_threshold, device=self._settings.resolve_diarization_device())
                
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
                    self._settings.gemini_model
                    if selected == "gemini"
                    else self._settings.model_name
                ),
                device=self._settings.resolve_device(),
                compute_type=self._settings.resolve_compute_type(),
                cpu_threads=self._settings.resolve_cpu_threads(),
                device_index=self._settings.resolve_device_index(),
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
        import json
        import re

        from google import genai

        from audiobench.output.text import TextFormatter

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

    def _spawn_auto_naming(self, tx_id: int, file_path: str, transcript: Transcript) -> None:
        """Spawn a background thread to generate a semantic title and rename the file."""
        import threading

        def _rename() -> None:
            try:
                import re
                from pathlib import Path

                from google import genai

                from audiobench.core.db_session import get_session
                from audiobench.storage.models import AudioFileRecord, TranscriptionRecord

                # Get first ~5 minutes of text to generate title
                intro_segments = [s for s in transcript.segments if s.end <= 300][:50]
                intro_text = " ".join([s.text for s in intro_segments])

                if not intro_text.strip():
                    return

                prompt = (
                    "Based on the following transcript excerpt, generate a concise, highly semantic 3-5 word title for this audio file.\n"
                    "Do NOT use quotes, special characters, or prefixes like 'Title:'. Just the raw title text.\n\n"
                    f"{intro_text}"
                )

                new_title = ""
                try:
                    if self._settings.gemini_api_key:
                        client = genai.Client(api_key=self._settings.gemini_api_key)
                        response = client.models.generate_content(
                            model="gemini-2.5-pro", contents=prompt
                        )
                        new_title = response.text.strip()
                    else:
                        raise ValueError("No Gemini API key configured")
                except Exception as e:
                    logger.warning(
                        "Gemini auto-rename failed/unavailable (%s), falling back to Ollama", e
                    )
                    from audiobench.chat.providers.ollama_provider import OllamaClient

                    ollama = OllamaClient(
                        base_url=self._settings.ollama_base_url,
                        model=self._settings.clean_model,
                    )
                    if not ollama.is_available():
                        logger.warning("Ollama not available for fallback rename")
                        return

                    response = ollama.chat([{"role": "user", "content": prompt}], think=False)
                    new_title = response.get("content", "").strip()

                if not new_title:
                    # BOTH FAILED! Tag the file for later retry.
                    with get_session() as session:
                        tx_record = session.query(TranscriptionRecord).get(tx_id)
                        if tx_record:
                            audio_record = session.query(AudioFileRecord).get(
                                tx_record.audio_file_id
                            )
                            if audio_record:
                                import json

                                try:
                                    tags_list = (
                                        json.loads(audio_record.tags) if audio_record.tags else []
                                    )
                                except Exception:
                                    tags_list = []
                                if "pending_auto_rename" not in tags_list:
                                    tags_list.append("pending_auto_rename")
                                    audio_record.tags = json.dumps(tags_list)
                        session.commit()
                    logger.info(
                        "Auto-rename failed completely. Tagged #%d with pending_auto_rename", tx_id
                    )
                    return

                # Clean up title for filesystem
                new_title = re.sub(r"[^\w\s-]", " ", new_title).strip()
                new_title = re.sub(r"\s+", " ", new_title)

                if not new_title or len(new_title.split()) > 15:
                    logger.warning("Auto-rename generated invalid title: %s", new_title)
                    return

                old_path = Path(file_path)
                if not old_path.exists():
                    logger.warning("Original file missing, skipping rename: %s", old_path)
                    return

                new_filename = f"{new_title}{old_path.suffix}"
                new_path = old_path.parent / new_filename

                # Handle collisions
                counter = 1
                while new_path.exists() and new_path != old_path:
                    new_path = old_path.parent / f"{new_title}_{counter}{old_path.suffix}"
                    new_filename = new_path.name
                    counter += 1

                if new_path != old_path:
                    old_path.rename(new_path)

                    # Update DB
                    with get_session() as session:
                        tx_record = session.query(TranscriptionRecord).get(tx_id)
                        if tx_record:
                            audio_record = session.query(AudioFileRecord).get(
                                tx_record.audio_file_id
                            )
                            if audio_record:
                                audio_record.file_name = new_filename
                                audio_record.file_path = str(new_path)
                        session.commit()

                    logger.info("Auto-renamed file to %s", new_filename)
            except Exception as e:
                logger.warning("Background auto-rename failed for #%d: %s", tx_id, e)

        thread = threading.Thread(target=_rename, name=f"rename-{tx_id}", daemon=True)
        thread.start()
        logger.info("Spawned background auto-rename thread for #%d", tx_id)

    def _reconstruct_transcript(self, data: dict, metadata: AudioMetadata) -> Transcript:
        """Reconstruct a Transcript from cached DB data."""
        from audiobench.transcribe.transcription_result import Segment

        segments = [
            Segment(
                id=s["index"],
                text=s["text"],
                start=s["start"],
                end=s["end"],
                speaker=s.get("speaker"),
            )
            for s in data.get("segments", [])
        ]
        return Transcript(
            segments=segments,
            language=data.get("language", "en"),
            language_probability=data.get("language_probability", 0.0),
            audio=metadata,
            duration_seconds=data.get("duration", 0.0),
            engine=data.get("engine", "faster-whisper"),
            model_name=data.get("model", "large-v3-turbo"),
        )
