"""YouTube fetcher module using yt-dlp."""

import json
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from audiobench.core.logger_factory import get_logger
from audiobench.cli.display.theme import console
from audiobench.core.settings import get_settings
from audiobench.storage.models import AudioFileRecord, JobQueueItem

logger = get_logger("youtube.fetcher")


@dataclass
class VideoMeta:
    """Extracted YouTube video metadata."""

    video_id: str
    title: str
    uploader: str
    duration: float
    upload_date: str | None = None
    description: str = ""


def extract_video_id(url_or_id: str) -> str:
    """Extract an 11-character video ID from a URL or return it if it's already an ID."""
    if len(url_or_id) == 11 and re.match(r"^[a-zA-Z0-9_-]{11}$", url_or_id):
        return url_or_id

    # regex to handle youtu.be, youtube.com/watch?v=, youtube.com/shorts/, etc.
    match = re.search(r"(?:v=|/)([0-9A-Za-z_-]{11}).*", url_or_id)
    if match:
        return match.group(1)

    raise ValueError(f"Could not extract video ID from '{url_or_id}'")


def get_video_metadata(video_id: str) -> VideoMeta:
    """Fetch video metadata using yt-dlp --dump-json."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = ["yt-dlp", "--dump-json", "--no-warnings", "--playlist-items", "1", url]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)

    return VideoMeta(
        video_id=video_id,
        title=data.get("title", "Unknown Title"),
        uploader=data.get("uploader", "Unknown Channel"),
        duration=data.get("duration", 0.0),
        upload_date=data.get("upload_date"),
        description=data.get("description", ""),
    )


def download_audio(video_id: str, output_dir: Path, progress_cb: Any = None) -> Path:
    """Download audio from a YouTube video and convert to .m4a.
    
    This operation is non-transactional and may leave a partial .tmp file if interrupted.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    # We use a naming template. yt-dlp will replace %(ext)s with m4a after FFmpeg processing.
    outtmpl = str(output_dir / "%(uploader)s - %(title)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "--format", "bestaudio/best",
        "--extract-audio",
        "--audio-format", "m4a",
        "--output", outtmpl,
        "--newline",  # CRITICAL: Forces yt-dlp to output progress lines separated by newlines
        "--no-warnings",
        url
    ]

    import re
    from rich.progress import Progress, TextColumn, BarColumn

    final_path = None

    with Progress(
        TextColumn("[bold blue]{task.description}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        console=console,
        expand=True,
    ) as progress:
        task_id = progress.add_task("Downloading", total=100.0)
        
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        
        # Regex to parse standard yt-dlp progress line:
        # [download]  15.3% of ~  20.00MiB at    1.50MiB/s ETA 00:15
        progress_re = re.compile(r"\[download\]\s+(?P<percent>[0-9\.]+)")
        # Regex to capture the final filename
        dest_re = re.compile(r"\[(?:ExtractAudio|download)\] Destination:\s+(.+\.m4a)")

        for line in process.stdout:
            # Check for progress percentage
            p_match = progress_re.search(line)
            if p_match:
                pct = float(p_match.group("percent"))
                progress.update(task_id, completed=pct)
            
            # Check for final destination
            d_match = dest_re.search(line)
            if d_match:
                final_path = d_match.group(1)

        process.wait()

    if process.returncode != 0:
        raise RuntimeError(f"Failed to download video {video_id}")

    if final_path:
        expected_path = Path(final_path)
    else:
        # Fallback: find the newest .m4a file in output_dir
        m4a_files = list(output_dir.glob("*.m4a"))
        if not m4a_files:
            raise FileNotFoundError(f"Failed to find any downloaded .m4a files in {output_dir}.")
        expected_path = max(m4a_files, key=lambda p: p.stat().st_mtime)
        
    if not expected_path.exists():
        raise FileNotFoundError(f"Expected to find downloaded file at {expected_path}, but it was missing.")
        
    return expected_path


def fetch_and_register(video_id: str, session: Any) -> tuple[AudioFileRecord | None, JobQueueItem | None]:
    """Fetch a YouTube video and queue it for transcription.
    
    Atomic safety guarantees:
    1. Uniqueness check is performed first.
    2. File download is performed (non-transactional, may leave artifacts).
    3. Both AudioFileRecord and JobQueueItem are inserted in a single DB commit.
       This means if the process crashes mid-download, no orphaned DB rows are created.
       
    Note on race conditions: This function is safe for single-user CLI usage. Under heavy
    concurrent usage, a race exists between the uniqueness check and the final commit,
    which could result in duplicate downloads followed by a UNIQUE constraint failure.
    """
    # 1. Uniqueness Check
    existing = session.query(AudioFileRecord).filter_by(youtube_video_id=video_id).first()
    if existing:
        return existing, None
        
    settings = get_settings()
    output_dir = settings.data_dir / "library" / "youtube"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Pre-fetch metadata so we can construct a nice title or abort early if private
    meta = get_video_metadata(video_id)
    
    # 2. File Download (Non-transactional)
    file_path = download_audio(video_id, output_dir)
    
    # 3. Transactional Insert
    try:
        audio_record = AudioFileRecord(
            file_path=str(file_path),
            file_name=file_path.name,
            file_size_bytes=file_path.stat().st_size,
            format=file_path.suffix.lstrip("."),
            duration_seconds=meta.duration,
            youtube_video_id=video_id,
            tags='["youtube"]',
        )
        session.add(audio_record)
        session.flush() # Get the ID
        
        job_record = JobQueueItem(
            file_path=str(file_path),
            status="pending"
        )
        session.add(job_record)
        
        session.commit()
        return audio_record, job_record
    except Exception as e:
        session.rollback()
        # Clean up the file if the DB insert failed
        if file_path.exists():
            file_path.unlink()
        raise e
