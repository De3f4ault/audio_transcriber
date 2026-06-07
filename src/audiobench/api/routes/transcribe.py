import asyncio
import json
import os
import shutil

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile
from sse_starlette.sse import EventSourceResponse

from audiobench.api.jobs import job_manager
from audiobench.core.settings import get_settings
from audiobench.transcribe.engines.engine_registry import create_engine

router = APIRouter()


def process_audio(job_id: str, audio_path: str, engine_name: str, language: str = None):
    """Synchronous CPU-bound task run in FastAPI's background threadpool."""
    job = job_manager.get_job(job_id)
    if not job:
        return

    job.status = "processing"
    settings = get_settings()

    try:
        # engine_registry.create_engine calls load_model automatically
        model_name = settings.gemini_model if engine_name == "gemini" else settings.model_name
        engine = create_engine(
            engine_name=engine_name,
            model_name=model_name,
            device=settings.resolve_device(),
            compute_type=settings.resolve_compute_type(),
            cpu_threads=settings.resolve_cpu_threads(),
            device_index=settings.resolve_device_index(),
        )
    except Exception as e:
        job.status = "error"
        job.error = f"Failed to create engine: {e}"
        return

    # Setup callbacks for live SSE streaming and cancellation
    def on_progress(percent: float):
        if job.is_cancelled:
            raise Exception("Job cancelled by user")
        job.progress = percent

    def on_segment(segment):
        if job.is_cancelled:
            raise Exception("Job cancelled by user")
        job.segments.append(segment.model_dump())

    try:
        transcript = engine.transcribe(
            audio_path,
            language=language,
            progress_callback=on_progress,
            on_segment=on_segment,
        )

        job.status = "completed"
        job.result = transcript.model_dump()
        job.progress = 100.0

    except Exception as e:
        if job.is_cancelled:
            job.status = "cancelled"
        else:
            job.status = "error"
            job.error = str(e)
    finally:
        # Clean up temp file
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except OSError:
                pass


@router.post("/")
async def upload_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    engine: str = Form("whisper"),
    language: str = Form(None),
):
    """Upload an audio file and start background transcription."""
    settings = get_settings()
    settings.ensure_dirs()

    # Save file in chunks to prevent memory spikes for huge files
    temp_path = settings.data_dir / f"temp_upload_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Create tracking job
    job = job_manager.create_job()

    # Launch synchronous task in background threadpool (prevents event loop blocking)
    background_tasks.add_task(process_audio, job.id, str(temp_path), engine, language)

    return {"job_id": job.id}


@router.get("/stream/{job_id}")
async def stream_job(request: Request, job_id: str):
    """Server-Sent Events (SSE) stream for live job progress and segments."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        last_seg_idx = 0
        last_progress = -1.0

        while True:
            if await request.is_disconnected():
                break

            # Yield any new segments
            while last_seg_idx < len(job.segments):
                seg = job.segments[last_seg_idx]
                yield {"event": "segment", "data": json.dumps(seg)}
                last_seg_idx += 1

            # Yield progress if changed
            if job.progress != last_progress:
                yield {"event": "progress", "data": json.dumps({"progress": job.progress})}
                last_progress = job.progress

            # Yield final status if finished
            if job.status in ["completed", "error", "cancelled"]:
                yield {
                    "event": "status",
                    "data": json.dumps(
                        {"status": job.status, "result": job.result, "error": job.error}
                    ),
                }
                break

            await asyncio.sleep(0.5)

    return EventSourceResponse(event_generator())


@router.delete("/{job_id}")
def cancel_job(job_id: str):
    """Cancel a running transcription job."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    job.is_cancelled = True
    return {"status": "cancelled"}
