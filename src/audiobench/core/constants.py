"""Centralized constants for AudioBench settings, validation, and mappings."""

# --- Whisper Engine Constants ---
WHISPER_MODELS = {"tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"}

# --- Hardware Constants ---
COMPUTE_DEVICES = {"auto", "cpu", "cuda"}
COMPUTE_TYPES = {"int8", "float16", "float32"}

# --- Output & Formatting ---
OUTPUT_FORMATS = {"txt", "srt", "vtt", "json"}

# --- Application Presets ---
SPEED_PRESETS = {"fast", "balanced", "accurate"}

# --- Audio Processing Constants ---
# Map common audio extensions to MIME types.
MIME_MAP = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".aac": "audio/aac",
    ".wma": "audio/x-ms-wma",
    ".webm": "audio/webm",
}


def get_mime(suffix: str) -> str:
    """Resolve MIME type from file extension."""
    return MIME_MAP.get(suffix.lower(), "audio/mpeg")
