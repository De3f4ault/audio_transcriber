"""Speaker diarization module — identify who is speaking.

Provides:
    PyannoteDiarizer — pyannote.audio-based speaker identification
"""

from src.audiobench.diarization.engine import PyannoteDiarizer

__all__ = ["PyannoteDiarizer"]
