"""Core exception hierarchy for AudioBench."""

from __future__ import annotations

# Import base exception to maintain backwards compatibility with core.error_types
from audiobench.core.error_types import AudioBenchError


class DaemonUnavailableError(AudioBenchError):
    """Daemon socket not reachable."""

    def __init__(
        self, message: str = "Daemon socket not reachable", details: str | None = None
    ) -> None:
        super().__init__(message, details)


class DaemonTimeoutError(AudioBenchError):
    """Daemon did not respond within configured timeout."""

    def __init__(
        self, message: str = "Daemon did not respond within timeout", details: str | None = None
    ) -> None:
        super().__init__(message, details)


class ExpressionNotFoundError(AudioBenchError):
    """SQLite lookup returned None."""

    def __init__(self, expression_id: int) -> None:
        super().__init__(
            message=f"Expression not found: {expression_id}",
            details=f"No expression record exists with ID {expression_id}",
        )


class EmbeddingError(AudioBenchError):
    """Model encode failure."""

    def __init__(self, reason: str) -> None:
        super().__init__(message="Embedding generation failed", details=reason)


class ChunkingError(AudioBenchError):
    """Chunking pipeline failure."""

    def __init__(self, reason: str) -> None:
        super().__init__(message="Chunking pipeline failed", details=reason)


class MigrationError(AudioBenchError):
    """Schema migration failed or was not idempotent."""

    def __init__(self, reason: str) -> None:
        super().__init__(message="Database migration failed", details=reason)


class SummaryGenerationError(AudioBenchError):
    """Ollama call failed during session summary."""

    def __init__(self, reason: str) -> None:
        super().__init__(message="Summary generation failed", details=reason)
