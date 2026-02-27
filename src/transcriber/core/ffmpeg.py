"""FFmpeg/FFprobe audio loader -- direct subprocess wrapper.

Replaces pydub with direct ffmpeg calls for:
- Instant metadata via ffprobe (no file loading)
- Piped conversion to float32 numpy (no temp files)
- Optional audio filter graphs (noise reduction, normalization)

System requirements: ffmpeg and ffprobe on PATH.

Usage:
    from src.transcriber.core.ffmpeg import AudioLoader

    with AudioLoader() as loader:
        wav_path, metadata = loader.load("/path/to/meeting.m4a")

    # Metadata only (instant):
    info = probe("/path/to/meeting.m4a")

    # With noise reduction filters:
    with AudioLoader() as loader:
        wav_path, metadata = loader.load(
            "/path/to/noisy.m4a",
            filters=["highpass=f=200", "afftdn=nf=-25", "dynaudnorm=p=0.9"],
        )
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.transcriber.config.logging_config import get_logger
from src.transcriber.core.exceptions import AudioLoadError, UnsupportedFormatError
from src.transcriber.core.models import AudioMetadata

logger = get_logger("core.ffmpeg")

# Default filter preset for --enhance
ENHANCE_FILTERS = [
    "highpass=f=200",  # remove low-frequency rumble (HVAC, traffic)
    "afftdn=nf=-25",  # adaptive noise reduction
    "dynaudnorm=p=0.9",  # dynamic volume normalization
]

# Formats ffmpeg can handle
SUPPORTED_AUDIO_FORMATS = {
    "m4a",
    "mp3",
    "wav",
    "flac",
    "ogg",
    "aac",
    "wma",
    "opus",
    "aiff",
    "webm",
    "amr",
    "oga",
    "spx",
}

SUPPORTED_VIDEO_FORMATS = {
    "mp4",
    "mkv",
    "avi",
    "mov",
    "m4v",
    "webm",
    "flv",
    "wmv",
}

ALL_SUPPORTED_FORMATS = SUPPORTED_AUDIO_FORMATS | SUPPORTED_VIDEO_FORMATS


def _check_ffmpeg() -> None:
    """Verify ffmpeg and ffprobe are available on PATH."""
    if not shutil.which("ffmpeg"):
        raise AudioLoadError(
            "ffmpeg",
            "ffmpeg not found on PATH. Install it: https://ffmpeg.org/download.html",
        )
    if not shutil.which("ffprobe"):
        raise AudioLoadError(
            "ffprobe",
            "ffprobe not found on PATH. Install it: https://ffmpeg.org/download.html",
        )


@dataclass
class AudioInfo:
    """Metadata from ffprobe."""

    duration: float  # seconds
    sample_rate: int
    channels: int
    codec: str
    bitrate: int  # bits per second (0 if unknown)
    format_name: str  # container format


def probe(file_path: str | Path) -> AudioInfo:
    """Get audio metadata via ffprobe. Instant, no file loading.

    Args:
        file_path: Path to audio/video file.

    Returns:
        AudioInfo with duration, sample rate, channels, codec, bitrate.

    Raises:
        AudioLoadError: If ffprobe fails or no audio stream found.
    """
    _check_ffmpeg()
    file_path = str(file_path)

    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        "-select_streams",
        "a:0",
        file_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise AudioLoadError(file_path, f"ffprobe failed: {result.stderr.strip()}")

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise AudioLoadError(file_path, f"ffprobe returned invalid JSON: {e}") from e

    streams = data.get("streams", [])
    if not streams:
        raise AudioLoadError(file_path, "No audio stream found")

    stream = streams[0]
    fmt = data.get("format", {})

    # Duration: prefer stream duration, fall back to format duration
    duration = float(stream.get("duration") or fmt.get("duration") or 0)

    return AudioInfo(
        duration=duration,
        sample_rate=int(stream.get("sample_rate", 0)),
        channels=int(stream.get("channels", 0)),
        codec=stream.get("codec_name", "unknown"),
        bitrate=int(stream.get("bit_rate") or fmt.get("bit_rate") or 0),
        format_name=fmt.get("format_name", "unknown"),
    )


def load_as_numpy(
    file_path: str | Path,
    filters: list[str] | None = None,
    target_sr: int = 16000,
    target_channels: int = 1,
) -> np.ndarray:
    """Convert any audio/video to a float32 numpy array via ffmpeg pipe.

    No temp files. Pipes ffmpeg stdout directly into memory.

    Args:
        file_path: Path to audio/video file.
        filters: Optional ffmpeg audio filter list (joined with ',').
        target_sr: Target sample rate (default: 16000 for Whisper).
        target_channels: Target channel count (default: 1 mono).

    Returns:
        float32 numpy array, normalized to [-1.0, 1.0].

    Raises:
        AudioLoadError: If ffmpeg fails.
    """
    _check_ffmpeg()
    file_path = str(file_path)

    cmd = [
        "ffmpeg",
        "-i",
        file_path,
        "-vn",  # strip video
        "-f",
        "s16le",  # raw PCM 16-bit little-endian
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(target_sr),
        "-ac",
        str(target_channels),
    ]

    if filters:
        cmd += ["-af", ",".join(filters)]

    cmd.append("-")  # output to stdout (pipe)

    logger.info("ffmpeg command: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()
        # Extract the last meaningful line from ffmpeg stderr
        lines = [line for line in stderr.splitlines() if line.strip()]
        error_msg = lines[-1] if lines else "unknown error"
        raise AudioLoadError(file_path, f"ffmpeg conversion failed: {error_msg}")

    if not result.stdout:
        raise AudioLoadError(file_path, "ffmpeg produced no output")

    # Convert raw PCM bytes to float32 numpy array
    audio = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0

    logger.info(
        "Loaded %d samples (%.1fs at %dHz)",
        len(audio),
        len(audio) / target_sr,
        target_sr,
    )

    return audio


def embed_subtitles(
    video_path: str | Path,
    subtitle_path: str | Path,
    output_path: str | Path,
    hard_burn: bool = False,
) -> Path:
    """Embed subtitles into a video file using ffmpeg.

    Args:
        video_path: Path to source video file.
        subtitle_path: Path to SRT or VTT subtitle file.
        output_path: Path for output video file.
        hard_burn: If True, render subtitles into the video pixels (permanent).
                   If False, add as a selectable subtitle track (soft embed).

    Returns:
        Path to the output video file.

    Raises:
        AudioLoadError: If ffmpeg fails or input files are invalid.
    """
    _check_ffmpeg()
    video_path = Path(video_path).resolve()
    subtitle_path = Path(subtitle_path).resolve()
    output_path = Path(output_path).resolve()

    if not video_path.exists():
        raise AudioLoadError(str(video_path), "Video file does not exist")
    if not subtitle_path.exists():
        raise AudioLoadError(str(subtitle_path), "Subtitle file does not exist")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if hard_burn:
        # Hard burn: render subtitles into video pixels (permanent, any container)
        # Requires re-encoding the video stream
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"subtitles={str(subtitle_path)}",
            "-c:a",
            "copy",  # copy audio as-is
            str(output_path),
        ]
    else:
        # Soft embed: mux subtitle track into container (selectable, no re-encode)
        # Works with mp4 (mov_text), mkv (srt/ass), webm (webvtt)
        ext = output_path.suffix.lstrip(".").lower()
        sub_codec = "mov_text" if ext == "mp4" else "srt"

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(subtitle_path),
            "-c",
            "copy",  # copy all streams
            "-c:s",
            sub_codec,  # subtitle codec for container
            str(output_path),
        ]

    logger.info("ffmpeg subtitle embed: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        lines = [line for line in result.stderr.splitlines() if line.strip()]
        error_msg = lines[-1] if lines else "unknown error"
        raise AudioLoadError(str(video_path), f"Subtitle embedding failed: {error_msg}")

    logger.info(
        "Subtitles embedded: %s → %s (mode=%s)",
        video_path.name,
        output_path.name,
        "hard_burn" if hard_burn else "soft_embed",
    )

    return output_path


class AudioLoader:
    """High-level audio loader with same interface as the old pydub-based one.

    Converts any supported format to 16kHz mono WAV for Whisper.
    Uses direct ffmpeg pipes -- no temp files for the conversion itself,
    but writes a temp WAV for the engine (faster-whisper expects a file path).

    Usage:
        with AudioLoader() as loader:
            wav_path, metadata = loader.load("meeting.m4a")
            wav_path, metadata = loader.load("noisy.mp3", filters=ENHANCE_FILTERS)
    """

    TARGET_SAMPLE_RATE = 16000
    TARGET_CHANNELS = 1

    def __init__(self, temp_dir: str | None = None) -> None:
        self._temp_dir = temp_dir
        self._temp_files: list[str] = []

    def load(
        self,
        file_path: str | Path,
        filters: list[str] | None = None,
    ) -> tuple[str, AudioMetadata]:
        """Load an audio file and convert to 16kHz mono WAV.

        Args:
            file_path: Path to audio/video file.
            filters: Optional ffmpeg audio filter list for preprocessing.

        Returns:
            Tuple of (path to WAV file, AudioMetadata).

        Raises:
            AudioLoadError: If the file cannot be loaded.
            UnsupportedFormatError: If the format is not supported.
        """
        file_path = Path(file_path).resolve()
        logger.info("Loading audio: %s", file_path.name)

        # Validate
        if not file_path.exists():
            raise AudioLoadError(str(file_path), "File does not exist")
        if not file_path.is_file():
            raise AudioLoadError(str(file_path), "Path is not a file")

        ext = file_path.suffix.lstrip(".").lower()
        if not ext:
            raise AudioLoadError(str(file_path), "File has no extension")
        if ext not in ALL_SUPPORTED_FORMATS:
            raise UnsupportedFormatError(str(file_path), ext)

        # Probe metadata (instant, no loading)
        info = probe(file_path)

        metadata = AudioMetadata(
            file_path=str(file_path),
            file_name=file_path.name,
            file_size_bytes=file_path.stat().st_size,
            format=ext,
            duration_seconds=round(info.duration, 3),
            sample_rate=info.sample_rate,
            channels=info.channels,
            file_hash=AudioMetadata.compute_file_hash(file_path),
        )

        logger.info(
            "Audio probed: %s, duration=%.1fs, codec=%s, rate=%dHz, channels=%d",
            file_path.name,
            info.duration,
            info.codec,
            info.sample_rate,
            info.channels,
        )

        # Convert to numpy via ffmpeg pipe
        audio_array = load_as_numpy(
            file_path,
            filters=filters,
            target_sr=self.TARGET_SAMPLE_RATE,
            target_channels=self.TARGET_CHANNELS,
        )

        # Write temp WAV for faster-whisper (it expects a file path)
        wav_path = self._write_wav(audio_array, file_path.stem)

        logger.info("Converted to 16kHz mono WAV: %s", wav_path)
        return wav_path, metadata

    def _write_wav(self, audio: np.ndarray, stem: str) -> str:
        """Write float32 numpy array to a temp WAV file."""
        import wave

        fd, wav_path = tempfile.mkstemp(
            prefix=f"transcriber_{stem}_",
            suffix=".wav",
            dir=self._temp_dir,
        )
        os.close(fd)

        # Convert float32 back to int16 for WAV
        pcm_data = (audio * 32767).astype(np.int16).tobytes()

        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(self.TARGET_CHANNELS)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.TARGET_SAMPLE_RATE)
            wf.writeframes(pcm_data)

        self._temp_files.append(wav_path)
        return wav_path

    def cleanup(self) -> None:
        """Remove temporary WAV files."""
        for path in self._temp_files:
            try:
                os.unlink(path)
                logger.debug("Cleaned up: %s", path)
            except OSError:
                pass
        self._temp_files.clear()

    @staticmethod
    def get_supported_formats() -> dict[str, set[str]]:
        """Return supported formats grouped by type."""
        return {
            "audio": SUPPORTED_AUDIO_FORMATS,
            "video": SUPPORTED_VIDEO_FORMATS,
        }

    def __enter__(self) -> AudioLoader:
        return self

    def __exit__(self, *args: object) -> None:
        self.cleanup()

    def __del__(self) -> None:
        self.cleanup()
