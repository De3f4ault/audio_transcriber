"""Embedding Engine — strictly typed entry point for vectorized embeddings.

Enforces task-specific prefix discipline for Nomic text embeddings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from audiobench.memory.singletons import get_primary_embedder


class EmbeddingEngine:
    """Entry point for generating vector embeddings.

    Enforces the 'search_document:' and 'search_query:' prefix discipline
    required by the nomic-embed-text-v1.5 model.
    """

    def embed_for_storage(self, text: str) -> np.ndarray:
        """Embed text intended for the storage layer.

        Args:
            text: The raw text content to embed.

        Returns:
            A 768-dimensional float32 numpy array.
        """
        prefixed_text = f"search_document: {text}"
        model = get_primary_embedder()
        # The encode method returns a numpy array by default
        return model.encode(prefixed_text)

    def embed_for_query(self, text: str) -> np.ndarray:
        """Embed text intended as a search query.

        Args:
            text: The user's query string.

        Returns:
            A 768-dimensional float32 numpy array.
        """
        prefixed_text = f"search_query: {text}"
        model = get_primary_embedder()
        return model.encode(prefixed_text)
