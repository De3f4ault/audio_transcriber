"""Tests for memory enums."""

from audiobench.memory.enums import (
    DaemonCmd,
    InferenceStatus,
    RelationType,
    SessionType,
    SourceType,
)


def test_source_type_values():
    assert SourceType.AUDIO_TRANSCRIPT.value == "audio_transcript"
    assert SourceType.TRANSCRIPT_SEGMENT.value == "transcript_segment"
    assert SourceType.CHAT_MESSAGE.value == "chat_message"
    assert SourceType.ASK_QUERY.value == "ask_query"
    assert SourceType.ASK_ANSWER.value == "ask_answer"
    assert SourceType.SESSION_SUMMARY.value == "session_summary"
    assert SourceType.DRIFT_PHASE.value == "drift_phase"
    assert SourceType.OPEN_THREAD.value == "open_thread"
    assert SourceType.BOOKMARK_NOTE.value == "bookmark_note"
    assert SourceType.CHAPTER_SUMMARY.value == "chapter_summary"
    assert SourceType.SYSTEM_INFERENCE.value == "system_inference"
    assert SourceType.USER_CORRECTION.value == "user_correction"

    # String-to-value roundtrip
    assert SourceType("audio_transcript") == SourceType.AUDIO_TRANSCRIPT


def test_relation_type_values():
    assert RelationType.SOURCE.value == "source"
    assert RelationType.SEMANTIC.value == "semantic"
    assert RelationType.TEMPORAL.value == "temporal"
    assert RelationType.THEMATIC.value == "thematic"
    assert RelationType.EXPLICIT.value == "explicit"
    assert RelationType.CORRECTS.value == "corrects"
    assert RelationType.ELABORATES.value == "elaborates"
    assert RelationType.RESUMES.value == "resumes"
    assert RelationType.CONTRADICTS.value == "contradicts"

    assert RelationType("source") == RelationType.SOURCE


def test_inference_status_values():
    assert InferenceStatus.ACTIVE.value == "active"
    assert InferenceStatus.CORRECTED.value == "corrected"
    assert InferenceStatus.DEPRECATED.value == "deprecated"

    assert InferenceStatus("active") == InferenceStatus.ACTIVE


def test_session_type_values():
    assert SessionType.CHAT.value == "chat"
    assert SessionType.ASK_LOG.value == "ask_log"
    assert SessionType.SYNTHESIS.value == "synthesis"

    assert SessionType("chat") == SessionType.CHAT


def test_daemon_cmd_values():
    assert DaemonCmd.PING.value == "ping"
    assert DaemonCmd.SEARCH.value == "search"
    assert DaemonCmd.EMBED.value == "embed"
    assert DaemonCmd.INFER.value == "infer"
    assert DaemonCmd.CHUNK.value == "chunk"
    assert DaemonCmd.STATUS.value == "status"
    assert DaemonCmd.REINDEX.value == "reindex"

    assert DaemonCmd("ping") == DaemonCmd.PING
