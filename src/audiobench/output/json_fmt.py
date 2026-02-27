"""JSON output formatter with full metadata."""

import json

from src.audiobench.core.models import Transcript
from src.audiobench.output.base import OutputFormatter


class JsonFormatter(OutputFormatter):
    """Format transcript as JSON with all metadata, timestamps, and words."""

    def format(self, transcript: Transcript) -> str:
        return transcript.model_dump_json(indent=2)

    @staticmethod
    def extension() -> str:
        return "json"
