"""Chapter detection and splitting logic."""

from .cue_parser import ChapterInfo, CueParser
from .detector import ChapterDetector
from .splitter import ChapterSplitter

__all__ = ["ChapterDetector", "CueParser", "ChapterInfo", "ChapterSplitter"]
