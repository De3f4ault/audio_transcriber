"""LocalRetrievalClient — in-process fallback when the daemon is unavailable.

Exposes the same interface as DaemonClient but loads all models directly
in the calling process. Used when auto-start fails or is explicitly disabled.
"""

from __future__ import annotations

from typing import Any

from audiobench.core.logger_factory import get_logger
from audiobench.daemon.protocol import ChunkResult, SearchResult
from audiobench.memory.enums import SourceType

logger = get_logger("daemon.local_client")


class LocalRetrievalClient:
    """Loads models in-process on first call (lazy, thread-safe via singletons)."""

    def __init__(self) -> None:
        self._store: Any = None

    def _get_store(self) -> Any:
        """Lazily initialise MemoryStore the first time it's needed."""
        if self._store is None:
            logger.warning(
                "Daemon unavailable — loading retrieval models in-process (cold start). "
                "Consider running `audiobench daemon start` for faster responses."
            )
            from audiobench.memory.memory_store import MemoryStore
            from audiobench.memory.singletons import pre_warm_retrieval_pipeline

            pre_warm_retrieval_pipeline()
            self._store = MemoryStore()
        return self._store

    # ------------------------------------------------------------------
    # RetrievalClient interface
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """Always True — we are the process."""
        return True

    def chunk(
        self, text: str, audio_file_id: int, diarized: bool, segments: list[dict] | None = None
    ) -> list[ChunkResult]:
        """Run text through the local chunking pipeline."""
        from audiobench.memory.chunking import content_aware_router

        chunks = content_aware_router(text, diarized_segments=segments if diarized else None)
        results = []
        for c in chunks:
            res: dict = {"content": c.content, "uuid": c.uuid, "tier": c.tier}
            if c.speaker:
                res["speaker"] = c.speaker
            results.append(res)  # type: ignore
        return results

    def search(
        self,
        query: str,
        top_k: int = 5,
        speaker_filter: str | None = None,
        hyde_document: str | None = None,
        use_bm25: bool = True,
        use_dense: bool = True,
        use_colbert: bool = True,
    ) -> list[SearchResult]:
        """Hybrid search over memory store."""
        return self._get_store().search(
            query=query,
            top_k=top_k,
            speaker_filter=speaker_filter,
            hyde_document=hyde_document,
            use_bm25=use_bm25,
            use_dense=use_dense,
            use_colbert=use_colbert,
        )

    def embed(
        self,
        expression_id: int,
        content: str,
        source_type: SourceType,
        speaker: str | None = None,
    ) -> None:
        """Embed and persist an expression to LanceDB."""
        self._get_store().write_node(
            expression_id=expression_id,
            content=content,
            source_type=source_type.value,
            speaker=speaker,
        )

    def delete(self, expression_id: int) -> None:
        """Remove an expression from LanceDB."""
        self._get_store().delete_node(expression_id)

    def status(self) -> dict:
        """Return store statistics."""
        store = self._get_store()
        return {
            "mode": "local",
            "uptime_seconds": 0.0,  # always 0 for local client
            "embedding_model_version": store.model_version,
            "total_nodes": store.count_nodes(),
        }

    def _get_cache(self) -> Any:
        """Lazily initialise QueryCacheStore."""
        if not hasattr(self, "_cache"):
            from audiobench.memory.memory_store import QueryCacheStore

            self._cache = QueryCacheStore()
        return self._cache

    def check_cache(self, query: str, distance_threshold: float = 0.05) -> dict | None:
        """Check semantic cache locally."""
        return self._get_cache().check_cache(query, distance_threshold)

    def write_cache(self, query: str, answer: str, hyde_document: str | None = None) -> None:
        """Write to semantic cache locally."""
        self._get_cache().write_cache(query, answer, hyde_document)

    def embed_query(self, text: str) -> list[float]:
        """Embed a query string using the daemon's warm Nomic model."""
        from audiobench.memory.singletons import get_primary_embedder, get_primary_inference_lock

        self._get_store()  # Ensure models are loaded
        text = str(text)[:12_000]
        prefixed = f"search_query: {text}"
        model = get_primary_embedder()
        with get_primary_inference_lock():
            return model.encode(prefixed).tolist()
