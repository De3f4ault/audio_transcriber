"""TypedDicts for daemon protocol messages."""

from __future__ import annotations

from typing import NotRequired, TypedDict


class EmbedArgs(TypedDict):
    expression_id: int
    content: str
    source_type: str
    speaker: NotRequired[str]


class SearchArgs(TypedDict):
    query: str
    top_k: int
    speaker_filter: NotRequired[str]


class InferArgs(TypedDict):
    expression_id: int
    content: str
    source_ids: list[int]


class CheckCacheArgs(TypedDict):
    query: str
    distance_threshold: float


class WriteCacheArgs(TypedDict):
    query: str
    answer: str
    hyde_document: NotRequired[str | None]


class ChunkArgs(TypedDict):
    text: str
    audio_file_id: int
    diarized: bool


class DaemonRequest(TypedDict):
    cmd: str
    args: dict
    request_id: str


class DaemonResponse(TypedDict):
    success: bool
    data: NotRequired[dict]
    error: NotRequired[str]
    request_id: str


class SearchResult(TypedDict):
    expression_id: int
    score: float
    content: str
    source_type: str
    speaker: NotRequired[str]


class ChunkResult(TypedDict):
    content: str
    uuid: str
    tier: int
    speaker: NotRequired[str]
