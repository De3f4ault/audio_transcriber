from unittest.mock import MagicMock, patch

from audiobench.memory.embedding_engine import EmbeddingEngine


def test_embedding_engine_storage_prefix():
    """Verify that embed_for_storage prepends 'search_document: '."""
    mock_model = MagicMock()
    mock_model.encode.return_value = [0.1, 0.2, 0.3]

    with patch("audiobench.memory.embedding_engine.get_primary_embedder", return_value=mock_model):
        engine = EmbeddingEngine()
        result = engine.embed_for_storage("test text")

        # Verify the prefix
        mock_model.encode.assert_called_once_with("search_document: test text")
        assert result == [0.1, 0.2, 0.3]


def test_embedding_engine_query_prefix():
    """Verify that embed_for_query prepends 'search_query: '."""
    mock_model = MagicMock()
    mock_model.encode.return_value = [0.4, 0.5, 0.6]

    with patch("audiobench.memory.embedding_engine.get_primary_embedder", return_value=mock_model):
        engine = EmbeddingEngine()
        result = engine.embed_for_query("test query")

        # Verify the prefix
        mock_model.encode.assert_called_once_with("search_query: test query")
        assert result == [0.4, 0.5, 0.6]
