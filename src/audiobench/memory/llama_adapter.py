"""LlamaIndex Adapter for the Nomic Embedding Engine.

Prevents LlamaIndex from loading a separate instance of the embedding model
and guarantees that LlamaIndex internals follow the task prefix discipline.
"""

from __future__ import annotations

import asyncio
from typing import Any

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr

from audiobench.memory.embedding_engine import EmbeddingEngine


class NomicEmbeddingAdapter(BaseEmbedding):
    """LlamaIndex compatible adapter for the primary embedding engine.

    Delegates to EmbeddingEngine to ensure 'search_document:' and
    'search_query:' prefixes are consistently applied.
    """

    _engine: EmbeddingEngine = PrivateAttr()

    def __init__(self, **kwargs: Any):
        """Initialize the adapter."""
        super().__init__(**kwargs)
        self._engine = EmbeddingEngine()

    def _get_query_embedding(self, query: str) -> list[float]:
        """Get query embedding."""
        return self._engine.embed_for_query(query).tolist()

    async def _aget_query_embedding(self, query: str) -> list[float]:
        """Asynchronously get query embedding."""
        # The underlying model is synchronous; run it in the default executor
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._get_query_embedding, query)

    def _get_text_embedding(self, text: str) -> list[float]:
        """Get text embedding."""
        return self._engine.embed_for_storage(text).tolist()

    async def _aget_text_embedding(self, text: str) -> list[float]:
        """Asynchronously get text embedding."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._get_text_embedding, text)

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get multiple text embeddings."""
        return [self._get_text_embedding(t) for t in texts]

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Asynchronously get multiple text embeddings."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._get_text_embeddings, texts)
