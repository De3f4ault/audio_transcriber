"""Base output formatter and formatter registry."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.audiobench.core.exceptions import OutputFormatError
from src.audiobench.core.models import Transcript

_FORMATTERS: dict[str, type[OutputFormatter]] = {}


class OutputFormatter(ABC):
    """Abstract base for output formatters."""

    @abstractmethod
    def format(self, transcript: Transcript) -> str:
        """Format a transcript into a string output."""

    @staticmethod
    @abstractmethod
    def extension() -> str:
        """File extension for this format (without dot)."""


def register_formatter(name: str, cls: type[OutputFormatter]) -> None:
    _FORMATTERS[name] = cls


def get_formatter(name: str) -> OutputFormatter:
    """Get a formatter instance by name."""
    _ensure_registered()
    if name not in _FORMATTERS:
        raise OutputFormatError(name, f"Unknown format. Available: {', '.join(_FORMATTERS.keys())}")
    return _FORMATTERS[name]()


def _ensure_registered() -> None:
    if _FORMATTERS:
        return
    from src.audiobench.output.text import TextFormatter
    from src.audiobench.output.srt import SrtFormatter
    from src.audiobench.output.vtt import VttFormatter
    from src.audiobench.output.json_fmt import JsonFormatter

    register_formatter("txt", TextFormatter)
    register_formatter("srt", SrtFormatter)
    register_formatter("vtt", VttFormatter)
    register_formatter("json", JsonFormatter)
