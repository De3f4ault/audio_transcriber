"""Enums for the AudioBench unified namespace."""

from __future__ import annotations

from enum import Enum


class SourceType(str, Enum):
    AUDIO_TRANSCRIPT = "audio_transcript"
    TRANSCRIPT_SEGMENT = "transcript_segment"
    CHAT_MESSAGE = "chat_message"
    ASK_QUERY = "ask_query"
    ASK_ANSWER = "ask_answer"
    SESSION_SUMMARY = "session_summary"
    DRIFT_PHASE = "drift_phase"
    OPEN_THREAD = "open_thread"
    BOOKMARK_NOTE = "bookmark_note"
    CHAPTER_SUMMARY = "chapter_summary"
    SYSTEM_INFERENCE = "system_inference"
    USER_CORRECTION = "user_correction"
    JOURNAL_ENTRY = "journal_entry"


class RelationType(str, Enum):
    SOURCE = "source"
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"
    THEMATIC = "thematic"
    EXPLICIT = "explicit"
    CORRECTS = "corrects"
    ELABORATES = "elaborates"
    RESUMES = "resumes"
    CONTRADICTS = "contradicts"


class InferenceStatus(str, Enum):
    ACTIVE = "active"
    CORRECTED = "corrected"
    DEPRECATED = "deprecated"


class SessionType(str, Enum):
    CHAT = "chat"
    ASK_LOG = "ask_log"
    SYNTHESIS = "synthesis"


class DaemonCmd(str, Enum):
    PING = "ping"
    SEARCH = "search"
    EMBED = "embed"
    INFER = "infer"
    CHUNK = "chunk"
    STATUS = "status"
    REINDEX = "reindex"
