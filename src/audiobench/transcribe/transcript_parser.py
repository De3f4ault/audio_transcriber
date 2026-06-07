"""Transcript parser for reverse import."""

import json
import re
from pathlib import Path

from audiobench.transcribe.transcription_result import Transcript, Segment, Word, AudioMetadata

class UnsupportedSchemaVersion(Exception):
    pass

class HashMismatchError(Exception):
    pass

SKIP_TRANSCRIPT = {"text", "word_count", "segment_count", "chapters", "audio", "segments"}
SKIP_SEGMENT = {"duration", "word_count"}
SKIP_WORD = {"duration", "midpoint"}

def parse_transcript_file(path: Path) -> tuple[Transcript, dict | None]:
    """Parse a transcript file. Returns (Transcript, raw_audio_dict | None)."""
    ext = path.suffix.lower()
    if ext == ".json":
        return _parse_json(path)
    elif ext == ".srt":
        return _parse_srt(path), None
    elif ext == ".txt":
        return _parse_txt(path), None
    else:
        raise ValueError(f"Unsupported format: {ext}")

def _parse_json(path: Path) -> tuple[Transcript, dict | None]:
    data = json.loads(path.read_text(encoding="utf-8"))
    
    version = data.get("schema_version", "1")
    if str(version) not in ("1",):
        raise UnsupportedSchemaVersion(f"schema_version '{version}' not supported")
    
    segments = []
    for raw_seg in data.get("segments", []):
        words = [
            Word(**{k: v for k, v in w.items() if k not in SKIP_WORD})
            for w in raw_seg.get("words", [])
        ]
        seg_data = {k: v for k, v in raw_seg.items() if k not in SKIP_SEGMENT | {"words"}}
        segments.append(Segment(**seg_data, words=words))
        
    tx_data = {k: v for k, v in data.items() if k not in SKIP_TRANSCRIPT}
    transcript = Transcript(**tx_data, segments=segments)
    
    return transcript, data.get("audio")

def _srt_time_to_seconds(time_str: str) -> float:
    # 00:12:47,123
    h, m, s = time_str.split(':')
    s, ms = s.split(',')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

def _parse_srt(path: Path) -> Transcript:
    content = path.read_text(encoding="utf-8")
    blocks = re.split(r'\n\s*\n', content.strip())
    
    segments = []
    
    for i, block in enumerate(blocks):
        lines = block.splitlines()
        if len(lines) >= 3:
            time_match = re.search(r'(\d+:\d+:\d+,\d+)\s*-->\s*(\d+:\d+:\d+,\d+)', lines[1])
            if time_match:
                start = _srt_time_to_seconds(time_match.group(1))
                end = _srt_time_to_seconds(time_match.group(2))
                text = " ".join(lines[2:])
                segments.append(Segment(id=i, text=text, start=start, end=end))
                
    duration = segments[-1].end if segments else 0.0
                
    return Transcript(
        segments=segments,
        language="en",
        language_probability=1.0,
        duration_seconds=duration,
        engine="external",
        model_name="unknown",
        speaker_map={},
        chapters=[]
    )

def _parse_txt(path: Path) -> Transcript:
    text = path.read_text(encoding="utf-8").strip()
    segments = []
    if text:
        segments.append(Segment(id=0, text=text, start=0.0, end=0.0))
        
    return Transcript(
        segments=segments,
        language="en",
        language_probability=1.0,
        duration_seconds=0.0,
        engine="external",
        model_name="unknown",
        speaker_map={},
        chapters=[]
    )

def build_audio_metadata(local_mp3_path: Path, raw_audio: dict | None) -> AudioMetadata:
    """
    Construct AudioMetadata for the local file.
    """
    local_hash = AudioMetadata.compute_file_hash(local_mp3_path)
    
    if raw_audio:
        json_hash = raw_audio.get("file_hash")
        if json_hash and json_hash != local_hash:
            raise HashMismatchError(
                f"Hash mismatch!\n"
                f"  JSON:  {json_hash}\n"
                f"  Local: {local_hash}\n"
                f"Wrong file selected."
            )
        
        duration = raw_audio.get("duration_seconds")
        if duration is None:
            # Fallback to ffprobe if JSON lacks metadata somehow
            from audiobench.transcribe.audio_converter import probe_metadata
            meta = probe_metadata(str(local_mp3_path))
            duration = meta.duration_seconds
            
        return AudioMetadata(
            file_path=str(local_mp3_path.absolute()),
            file_name=raw_audio.get("file_name", local_mp3_path.name),
            file_size_bytes=raw_audio.get("file_size_bytes", local_mp3_path.stat().st_size),
            format=raw_audio.get("format", local_mp3_path.suffix.lstrip(".")),
            duration_seconds=duration,
            sample_rate=raw_audio.get("sample_rate", 44100),
            channels=raw_audio.get("channels", 2),
            file_hash=local_hash,
        )
    else:
        # Complete fallback
        from audiobench.transcribe.audio_converter import probe_metadata
        meta = probe_metadata(str(local_mp3_path))
        return AudioMetadata(
            file_path=str(local_mp3_path.absolute()),
            file_name=local_mp3_path.name,
            file_size_bytes=local_mp3_path.stat().st_size,
            format=local_mp3_path.suffix.lstrip("."),
            duration_seconds=meta.duration_seconds,
            sample_rate=meta.sample_rate,
            channels=meta.channels,
            file_hash=local_hash,
        )
