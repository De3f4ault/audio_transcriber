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
    args: dict[str, Any]
    request_id: str


class DaemonError(TypedDict):
    code: str      # "MODEL_NOT_LOADED" | "INDEX_NOT_READY" | "INVALID_REQUEST" | "OPERATION_FAILED"
    message: str
    request_id: str


class DaemonResponse(TypedDict):
    status: str  # "ok" | "error" | "progress"
    success: NotRequired[bool]  # deprecated, use status="ok"
    data: NotRequired[dict[str, Any]]
    error: NotRequired[DaemonError | str]
    request_id: str


class ProgressFrame(TypedDict):
    status: str  # always "progress"
    step: str
    pct: float
    message: NotRequired[str]
    request_id: str


class PipelineStep(TypedDict):
    name: str
    args: dict[str, Any]


class PipelineArgs(TypedDict):
    steps: list[PipelineStep]


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


class OptimizeResult(TypedDict):
    tables_optimized: list[str]
    duration_seconds: float
    last_optimized_at: str
    bytes_freed: NotRequired[int]
