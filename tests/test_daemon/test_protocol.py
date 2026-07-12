"""Tests for daemon protocol TypedDicts."""

import json
from typing import cast

from audiobench.daemon.protocol import DaemonRequest, DaemonResponse, EmbedArgs


def test_daemon_request_serialization() -> None:
    args: EmbedArgs = {
        "expression_id": 1,
        "content": "hello world",
        "source_type": "audio_transcript",
    }
    req: DaemonRequest = {
        "cmd": "embed",
        "args": dict(args),  # type: ignore[misc]
        "request_id": "req-123",
    }

    # Serialization
    encoded = json.dumps(req)
    assert "embed" in encoded
    assert "audio_transcript" in encoded

    # Deserialization roundtrip
    decoded = cast(DaemonRequest, json.loads(encoded))
    assert decoded["cmd"] == "embed"
    assert decoded["request_id"] == "req-123"
    assert decoded["args"]["expression_id"] == 1


def test_daemon_response_serialization() -> None:
    resp: DaemonResponse = {"status": "ok", "success": True, "data": {"status": "ok"}, "request_id": "req-123"}

    encoded = json.dumps(resp)
    decoded = cast(DaemonResponse, json.loads(encoded))
    assert decoded["success"] is True
    assert decoded["status"] == "ok"
    assert decoded["data"]["status"] == "ok"
    assert decoded["request_id"] == "req-123"

def test_progress_frame_serialization() -> None:
    from audiobench.daemon.protocol import ProgressFrame
    frame: ProgressFrame = {"status": "progress", "step": "search", "pct": 0.5, "request_id": "req-1"}
    encoded = json.dumps(frame)
    decoded = cast(ProgressFrame, json.loads(encoded))
    assert decoded["status"] == "progress"
    assert decoded["step"] == "search"
