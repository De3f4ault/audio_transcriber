"""Focused entity abstraction for REPL context."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class FocusedEntity:
    """The root context for the current REPL session.

    A session is almost always focused on a file. The transcript ID
    is treated as a derived property of the focused file.
    """

    type: Literal["file", "transcript", "project", "work"]
    id: int  # AudioFileRecord.id, TranscriptionRecord.id, or WorkRecord.id
    label: str  # e.g., "meeting.mp4" or "Transcript #42"
    chapter_index: int | None = None
    chapter_title: str | None = None

    @property
    def display_label(self) -> str:
        """Returns the prompt string, appending the chapter if focused."""
        if self.chapter_title:
            return f"{self.label} › {self.chapter_title}"
        return self.label
