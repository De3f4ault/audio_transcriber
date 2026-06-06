"""Protocol interface for daemon clients."""

from __future__ import annotations

from typing import Protocol

from audiobench.daemon.protocol import ChunkResult, SearchResult
from audiobench.memory.enums import SourceType


class RetrievalClient(Protocol):
    """Shared interface for daemon and local retrieval clients."""

    def search(
        self,
        query: str,
        top_k: int,
        speaker_filter: str | None = None,
        hyde_document: str | None = None,
        use_bm25: bool = True,
        use_dense: bool = True,
        use_colbert: bool = True,
    ) -> list[SearchResult]: ...

    def embed(
        self, expression_id: int, content: str, source_type: SourceType, speaker: str | None
    ) -> None: ...

    def chunk(self, text: str, audio_file_id: int, diarized: bool) -> list[ChunkResult]: ...

    def check_cache(self, query: str, distance_threshold: float = 0.05) -> dict | None: ...

    def write_cache(self, query: str, answer: str, hyde_document: str | None = None) -> None: ...

    def ping(self) -> bool: ...
