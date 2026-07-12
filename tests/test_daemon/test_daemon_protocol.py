"""
Protocol-level tests for the daemon _dispatch async generator.

These tests exercise _dispatch() directly (no socket), but against the live
streaming protocol introduced in 1A. They replace the old synchronous
`json.loads(_dispatch(raw))` pattern with an asyncio helper that drains the
async generator and returns the final terminal frame.

The daemon does NOT need to be running — _dispatch() is called in-process.
"""

import asyncio
import json
from unittest.mock import MagicMock, patch

from audiobench.daemon.server import _dispatch


# ---------------------------------------------------------------------------
# Helper: drain an async generator, return all frames
# ---------------------------------------------------------------------------

async def _drain(raw: str) -> list[dict]:
    frames = []
    async for line in _dispatch(raw):
        frames.append(json.loads(line))
    return frames


def _run(raw: str) -> list[dict]:
    """Synchronous helper: run _drain in a fresh event loop."""
    return asyncio.run(_drain(raw))


def _terminal(frames: list[dict]) -> dict:
    """Return the last frame (ok or error terminal)."""
    assert frames, "No frames yielded by _dispatch"
    return frames[-1]


# ---------------------------------------------------------------------------
# 1. Ping contract
# ---------------------------------------------------------------------------

def test_daemon_ping_contract() -> None:
    """_dispatch('ping') yields a terminal ok frame with alive=True."""
    import audiobench.daemon.server as _srv
    with patch("audiobench.daemon.server._get_store") as mock_get_store, \
         patch.object(_srv, "_memory_store", new=object()):  # truthy sentinel
        mock_store = MagicMock()
        mock_store.model_version = "mock-model-v1"
        mock_get_store.return_value = mock_store

        request = {"cmd": "ping", "args": {}, "request_id": "test-req-123"}
        frames = _run(json.dumps(request))
        resp = _terminal(frames)

    assert resp["success"] is True
    assert resp["request_id"] == "test-req-123"
    assert resp["data"]["alive"] is True
    assert resp["data"]["embedding_model_version"] == "mock-model-v1"


# ---------------------------------------------------------------------------
# 2. Unknown command → error frame
# ---------------------------------------------------------------------------

def test_daemon_unknown_command() -> None:
    """_dispatch with unknown cmd yields an error frame (not a crash)."""
    request = {"cmd": "does_not_exist", "args": {}, "request_id": "test-err-404"}
    frames = _run(json.dumps(request))
    resp = _terminal(frames)

    assert resp["success"] is False
    assert resp["request_id"] == "test-err-404"
    # Error is now a structured dict per Standard 6; message contains the verb.
    error = resp.get("error", {})
    if isinstance(error, dict):
        assert "does_not_exist" in error.get("message", "") or \
               "Unknown command" in error.get("message", ""), \
               f"Expected command name in error message, got: {error}"
    else:
        # Fallback: some older frames embed error as a string
        assert "Unknown command" in str(error)


# ---------------------------------------------------------------------------
# 3. Search contract
# ---------------------------------------------------------------------------

def test_daemon_search_contract() -> None:
    """_dispatch('search') calls store.search with correct kwargs and returns results."""
    with patch("audiobench.daemon.server._get_store") as mock_get_store:
        mock_store = MagicMock()
        mock_store.search.return_value = [
            {"expression_id": 1, "score": 0.99, "content": "hello",
             "source_type": "transcription"}
        ]
        mock_get_store.return_value = mock_store

        request = {
            "cmd": "search",
            "args": {"query": "hello world", "top_k": 3},
            "request_id": "test-search-1",
        }
        frames = _run(json.dumps(request))
        resp = _terminal(frames)

    assert resp["success"] is True
    assert len(resp["data"]["results"]) == 1
    assert resp["data"]["results"][0]["expression_id"] == 1

    mock_store.search.assert_called_once_with(
        query="hello world",
        top_k=3,
        speaker_filter=None,
        audio_file_id=None,
        work_id=None,
        hyde_document=None,
        use_bm25=True,
        use_dense=True,
        use_colbert=True,
    )
