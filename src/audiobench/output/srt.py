"""SubRip (.srt) subtitle formatter."""

from audiobench.output.base import OutputFormatter
from audiobench.transcribe.transcription_result import Segment, Transcript, Word


def _format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _split_segment_by_speaker(seg: Segment) -> list[tuple[str | None, float, float, str]]:
    """Split a segment into (speaker, start, end, text) chunks by speaker turn.

    If the segment has no word-level speaker labels, returns the segment as-is.
    Each chunk represents a continuous run of words from the same speaker.
    """
    if not any(w.speaker for w in seg.words):
        return [(seg.speaker, seg.start, seg.end, seg.text)]

    chunks: list[tuple[str | None, float, float, str]] = []
    current_words: list[Word] = []
    current_speaker: str | None = None

    for word in seg.words:
        spk = word.speaker or seg.speaker
        if spk != current_speaker and current_words:
            # Flush current chunk
            chunks.append((
                current_speaker,
                current_words[0].start,
                current_words[-1].end,
                " ".join(w.word for w in current_words),
            ))
            current_words = []
        current_speaker = spk
        current_words.append(word)

    if current_words:
        chunks.append((
            current_speaker,
            current_words[0].start,
            current_words[-1].end,
            " ".join(w.word for w in current_words),
        ))

    return chunks


class SrtFormatter(OutputFormatter):
    """Format transcript as SubRip (.srt) subtitles.

    When word-level speaker labels are present, speaker changes within
    a segment generate separate numbered cue blocks with correct timestamps.
    """

    def format(self, transcript: Transcript) -> str:
        lines: list[str] = []
        cue_index = 1

        for seg in transcript.segments:
            chunks = _split_segment_by_speaker(seg)
            for speaker, start, end, text in chunks:
                lines.append(str(cue_index))
                lines.append(f"{_format_srt_time(start)} --> {_format_srt_time(end)}")
                if speaker:
                    text = f"[{speaker}] {text}"
                lines.append(text)
                lines.append("")
                cue_index += 1

        return "\n".join(lines)

    @staticmethod
    def extension() -> str:
        return "srt"
