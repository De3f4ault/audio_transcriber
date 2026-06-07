"""JSON output formatter with full metadata."""

from audiobench.output.base import OutputFormatter
from audiobench.transcribe.transcription_result import Transcript


class JsonFormatter(OutputFormatter):
    """Format transcript as JSON with all metadata, timestamps, and words."""

    def format(self, transcript: Transcript) -> str:
        import json
        data = json.loads(transcript.model_dump_json())
        data = {"schema_version": "1", **data}
        return json.dumps(data, indent=2, ensure_ascii=False)

    @staticmethod
    def extension() -> str:
        return "json"
