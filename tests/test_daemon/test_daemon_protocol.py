import json
from unittest.mock import MagicMock, patch

from audiobench.daemon.server import _dispatch


def test_daemon_ping_contract() -> None:
    """Verify the daemon dispatch returns a correct ping response."""
    # We must patch get_settings and get_store to bypass actual LanceDB
    with patch("audiobench.daemon.server._get_store") as mock_get_store:
        mock_store = MagicMock()
        mock_store.model_version = "mock-model-v1"
        mock_get_store.return_value = mock_store

        request = {"cmd": "ping", "args": {}, "request_id": "test-req-123"}

        raw_req = json.dumps(request)
        raw_resp = _dispatch(raw_req)

        resp = json.loads(raw_resp)
        assert resp["success"] is True
        assert resp["request_id"] == "test-req-123"
        assert resp["data"]["alive"] is True
        assert resp["data"]["embedding_model_version"] == "mock-model-v1"


def test_daemon_unknown_command() -> None:
    """Verify the daemon handles invalid commands properly."""
    request = {"cmd": "does_not_exist", "args": {}, "request_id": "test-err-404"}
    raw_req = json.dumps(request)
    raw_resp = _dispatch(raw_req)

    resp = json.loads(raw_resp)
    assert resp["success"] is False
    assert resp["request_id"] == "test-err-404"
    assert "Unknown command" in resp["error"]


def test_daemon_search_contract() -> None:
    """Verify search command parsing and response formatting."""
    with patch("audiobench.daemon.server._get_store") as mock_get_store:
        mock_store = MagicMock()
        mock_store.search.return_value = [
            {"expression_id": 1, "score": 0.99, "content": "hello", "source_type": "transcription"}
        ]
        mock_get_store.return_value = mock_store

        request = {
            "cmd": "search",
            "args": {"query": "hello world", "top_k": 3},
            "request_id": "test-search-1",
        }

        raw_req = json.dumps(request)
        raw_resp = _dispatch(raw_req)

        resp = json.loads(raw_resp)
        assert resp["success"] is True
        assert len(resp["data"]["results"]) == 1
        assert resp["data"]["results"][0]["expression_id"] == 1

        mock_store.search.assert_called_once_with(
            query="hello world",
            top_k=3,
            speaker_filter=None,
            hyde_document=None,
            use_bm25=True,
            use_dense=True,
            use_colbert=True,
        )
