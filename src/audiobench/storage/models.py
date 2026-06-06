"""SQLAlchemy ORM models for persisting transcription, chat, and bookmark data.

Tables:
    audio_files: Source audio file metadata + SHA-256 hash for dedup
    transcriptions: Transcription results linked to audio files
    segments: Individual segments within a transcription
    chat_conversations: Persistent AI chat sessions
    chat_messages: Individual messages within a chat conversation
    bookmarks: Timestamp markers and region annotations for audio files
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""

    pass


class AudioFileRecord(Base):
    """Persisted audio file metadata."""

    __tablename__ = "audio_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_name: Mapped[str] = mapped_column(String(256), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    format: Mapped[str] = mapped_column(String(16), default="unknown")
    duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    sample_rate: Mapped[int] = mapped_column(Integer, default=0)
    channels: Mapped[int] = mapped_column(Integer, default=0)
    file_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))
    tags: Mapped[str] = mapped_column(Text, default="[]")

    # Relationships
    chapters: Mapped[list[ChapterRecord]] = relationship(
        back_populates="audio_file",
        cascade="all, delete-orphan",
        order_by="ChapterRecord.chapter_index",
    )
    transcriptions: Mapped[list[TranscriptionRecord]] = relationship(
        back_populates="audio_file", cascade="all, delete-orphan"
    )
    bookmarks: Mapped[list[BookmarkRecord]] = relationship(
        back_populates="audio_file", cascade="all, delete-orphan"
    )

    @property
    def has_chapters(self) -> bool:
        """True if the audio file has any chapters."""
        return len(self.chapters) > 0

    def __repr__(self) -> str:
        return (
            f"<AudioFile(id={self.id}, name='{self.file_name}', "
            f"duration={self.duration_seconds:.1f}s)>"
        )


class ChapterRecord(Base):
    """Persisted chapter metadata for an audio file."""

    __tablename__ = "chapters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    audio_file_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("audio_files.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chapter_index: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(String(512), default="Untitled", nullable=False)
    start_time: Mapped[float] = mapped_column(Float, nullable=False)
    end_time: Mapped[float] = mapped_column(Float, nullable=False)
    transcription_status: Mapped[str] = mapped_column(String(20), default="pending")
    transcription_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("transcriptions.id", ondelete="SET NULL"), nullable=True
    )
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[str] = mapped_column(Text, default="[]")
    is_ghost: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))

    # Relationships
    audio_file: Mapped[AudioFileRecord] = relationship(back_populates="chapters")
    transcription: Mapped[TranscriptionRecord | None] = relationship()
    segments: Mapped[list[SegmentRecord]] = relationship(
        back_populates="chapter", cascade="all, delete-orphan"
    )
    bookmarks: Mapped[list[BookmarkRecord]] = relationship(
        back_populates="chapter", cascade="all, delete-orphan"
    )
    jobs: Mapped[list[JobRecord]] = relationship(
        back_populates="chapter", cascade="all, delete-orphan"
    )

    @property
    def is_real(self) -> bool:
        """True if this is a real chapter, false if it's a ghost chapter (start == end)."""
        return not bool(self.is_ghost)

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    @property
    def tags_list(self) -> list[str]:
        """Deserialise the JSON tags column to a Python list."""
        import json as _json

        try:
            return _json.loads(self.tags) if self.tags else []
        except Exception:
            return []

    def __repr__(self) -> str:
        return (
            f"<Chapter(id={self.id}, index={self.chapter_index}, title='{self.title[:30]}', "
            f"status='{self.transcription_status}')>"
        )


class TranscriptionRecord(Base):
    """Persisted transcription result."""

    __tablename__ = "transcriptions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    audio_file_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("audio_files.id"), nullable=True, index=True
    )
    source: Mapped[str] = mapped_column(String(20), default="file")
    file_name: Mapped[str] = mapped_column(String(256), default="", nullable=False)
    full_text: Mapped[str] = mapped_column(Text, default="")
    raw_text: Mapped[str] = mapped_column(Text, default="")
    language: Mapped[str] = mapped_column(String(10), default="en", index=True)
    language_probability: Mapped[float] = mapped_column(Float, default=0.0)
    engine: Mapped[str] = mapped_column(String(64), default="faster-whisper")
    model_name: Mapped[str] = mapped_column(String(64), default="large-v3-turbo")
    duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    word_count: Mapped[int] = mapped_column(Integer, default=0)
    segment_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(20), default="completed")
    speaker_map: Mapped[str] = mapped_column(Text, default="{}")
    refined_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, default=None)
    is_indexed: Mapped[int] = mapped_column(Integer, default=0, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), index=True
    )

    # Relationships
    audio_file: Mapped[AudioFileRecord] = relationship(back_populates="transcriptions")
    segments: Mapped[list[SegmentRecord]] = relationship(
        back_populates="transcription", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<Transcription(id={self.id}, lang='{self.language}', "
            f"words={self.word_count}, model='{self.model_name}')>"
        )


class SegmentRecord(Base):
    """Persisted segment within a transcription."""

    __tablename__ = "segments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    transcription_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("transcriptions.id"), nullable=False, index=True
    )
    segment_index: Mapped[int] = mapped_column(Integer, default=0)
    text: Mapped[str] = mapped_column(Text, default="")
    start_time: Mapped[float] = mapped_column(Float, default=0.0)
    end_time: Mapped[float] = mapped_column(Float, default=0.0)
    speaker: Mapped[str | None] = mapped_column(String(64), nullable=True)
    chapter_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("chapters.id", ondelete="SET NULL"), nullable=True, index=True
    )

    # Relationships
    transcription: Mapped[TranscriptionRecord] = relationship(back_populates="segments")
    chapter: Mapped[ChapterRecord | None] = relationship(back_populates="segments")

    def __repr__(self) -> str:
        return f"<Segment(id={self.id}, idx={self.segment_index}, text='{self.text[:30]}...')>"


class ChatConversation(Base):
    """A persistent AI chat conversation."""

    __tablename__ = "chat_conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(256), default="Untitled Chat")
    model_name: Mapped[str] = mapped_column(String(128), default="")
    session_type: Mapped[str] = mapped_column(String(64), default="chat")
    engine: Mapped[str] = mapped_column(String(64), default="ollama")
    transcript_ids: Mapped[str] = mapped_column(
        String(512), default="[]"
    )  # JSON list, e.g. "[3,5,7]"
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    # Relationships
    messages: Mapped[list[ChatMessage]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<ChatConversation(id={self.id}, title='{self.title}', messages={self.message_count})>"
        )


class ChatMessage(Base):
    """A single message in a chat conversation."""

    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("chat_conversations.id"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)  # system|user|assistant
    content: Mapped[str] = mapped_column(Text, default="")
    thinking: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))

    # Relationships
    conversation: Mapped[ChatConversation] = relationship(back_populates="messages")

    def __repr__(self) -> str:
        preview = self.content[:40] if self.content else ""
        return f"<ChatMessage(id={self.id}, role='{self.role}', text='{preview}...')>"


class BookmarkRecord(Base):
    """Persisted bookmark or region marker for an audio file.

    Point bookmarks have only `timestamp`; region markers also set
    `end_timestamp` to define a span.
    """

    __tablename__ = "bookmarks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    audio_file_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("audio_files.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    transcription_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("transcriptions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    end_timestamp: Mapped[float | None] = mapped_column(Float, nullable=True)
    name: Mapped[str] = mapped_column(String(512), default="Untitled")
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    bookmark_type: Mapped[str] = mapped_column(String(16), default="bookmark")
    color: Mapped[str | None] = mapped_column(String(16), nullable=True)
    chapter_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("chapters.id", ondelete="CASCADE"), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(UTC),
    )

    # Relationships
    audio_file: Mapped[AudioFileRecord] = relationship(back_populates="bookmarks")
    transcription: Mapped[TranscriptionRecord | None] = relationship()
    chapter: Mapped[ChapterRecord | None] = relationship(back_populates="bookmarks")

    @property
    def is_region(self) -> bool:
        """True if this bookmark defines a region (start + end)."""
        return self.end_timestamp is not None

    def __repr__(self) -> str:
        kind = "Region" if self.is_region else "Point"
        return f"<Bookmark(id={self.id}, {kind}, t={self.timestamp:.1f}, name='{self.name[:30]}')>"


class JobRecord(Base):
    """Persisted background job state."""

    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    command: Mapped[str] = mapped_column(Text, nullable=False)
    pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), default="running"
    )  # running, done, failed, cancelled
    log_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    events_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    audio_file: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    chapter_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("chapters.id", ondelete="CASCADE"), nullable=True
    )
    started_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    exit_code: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Relationships
    chapter: Mapped[ChapterRecord | None] = relationship(back_populates="jobs")

    def __repr__(self) -> str:
        return f"<Job(id={self.id}, status='{self.status}', cmd='{self.command[:30]}')>"


class ExpressionRecord(Base):
    """A semantic expression representing a unit of memory."""

    __tablename__ = "expressions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str | None] = mapped_column(
        String(64), index=True, nullable=True
    )
    source_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    session_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    session_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    speaker: Mapped[str | None] = mapped_column(String(64), nullable=True)
    inference_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    inference_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), index=True
    )

    # Relationships
    relations_from: Mapped[list[ExpressionRelation]] = relationship(
        "ExpressionRelation",
        foreign_keys="ExpressionRelation.from_expression_id",
        back_populates="source_expression",
        cascade="all, delete-orphan",
    )
    relations_to: Mapped[list[ExpressionRelation]] = relationship(
        "ExpressionRelation",
        foreign_keys="ExpressionRelation.to_expression_id",
        back_populates="target_expression",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<ExpressionRecord(id={self.id}, source_type='{self.source_type}', len={len(self.content)})>"


class ExpressionRelation(Base):
    """Directed relation between two semantic expressions."""

    __tablename__ = "expression_relations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    from_expression_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("expressions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    to_expression_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("expressions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    relation_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    created_by: Mapped[str] = mapped_column(String(64), default="system")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))

    # Relationships
    source_expression: Mapped[ExpressionRecord] = relationship(
        "ExpressionRecord", foreign_keys=[from_expression_id], back_populates="relations_from"
    )
    target_expression: Mapped[ExpressionRecord] = relationship(
        "ExpressionRecord", foreign_keys=[to_expression_id], back_populates="relations_to"
    )

    def __repr__(self) -> str:
        return f"<ExpressionRelation({self.from_expression_id} -> {self.to_expression_id}, type='{self.relation_type}')>"


class AskLog(Base):
    """Log of questions and answers for a specific audio file."""

    __tablename__ = "ask_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    audio_file_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("audio_files.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    entry_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )

    # Relationships
    entries: Mapped[list[AskEntry]] = relationship(
        "AskEntry", back_populates="log", cascade="all, delete-orphan"
    )
    audio_file: Mapped[AudioFileRecord] = relationship()

    def __repr__(self) -> str:
        return f"<AskLog(id={self.id}, audio_file_id={self.audio_file_id}, entries={self.entry_count})>"


class AskEntry(Base):
    """A single question and answer in an ask log."""

    __tablename__ = "ask_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    log_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("ask_logs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    question_expression_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("expressions.id", ondelete="SET NULL"), nullable=True
    )
    answer_expression_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("expressions.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))

    # Relationships
    log: Mapped[AskLog] = relationship("AskLog", back_populates="entries")

    def __repr__(self) -> str:
        return f"<AskEntry(id={self.id}, log_id={self.log_id})>"


class ConversationSummary(Base):
    """Semantic summary and insight extraction from a chat conversation."""

    __tablename__ = "conversation_summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("chat_conversations.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    narrative: Mapped[str] = mapped_column(Text, nullable=False)
    drift_phases: Mapped[str] = mapped_column(Text, default="[]")  # JSON
    key_insights: Mapped[str] = mapped_column(Text, default="[]")  # JSON
    open_threads: Mapped[str] = mapped_column(Text, default="[]")  # JSON
    refined_title: Mapped[str | None] = mapped_column(String(256), nullable=True)
    expression_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("expressions.id", ondelete="SET NULL"), nullable=True
    )
    generated_by: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))

    # Relationships
    conversation: Mapped[ChatConversation] = relationship()

    def __repr__(self) -> str:
        return f"<ConversationSummary(id={self.id}, conv_id={self.conversation_id})>"


class StagingCartItem(Base):
    """A persisted item in the user's transcription staging cart."""

    __tablename__ = "staging_cart"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    audio_file_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("audio_files.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    engine: Mapped[str] = mapped_column(String(64), default="gemini")
    model_name: Mapped[str] = mapped_column(String(64), default="large-v3-turbo")
    speed_preset: Mapped[str] = mapped_column(String(64), default="balanced")
    strategy: Mapped[str] = mapped_column(String(64), default="batch")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))

    # Relationships
    audio_file: Mapped[AudioFileRecord] = relationship()

    def __repr__(self) -> str:
        return f"<StagingCartItem(id={self.id}, audio_id={self.audio_file_id}, engine='{self.engine}')>"


class JobQueueItem(Base):
    """A persistent background job for sequential execution."""

    __tablename__ = "job_queue"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    engine: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    speed_preset: Mapped[str | None] = mapped_column(String(64), nullable=True)
    strategy: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), default="pending"
    )  # pending, processing, done, failed
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))

    def __repr__(self) -> str:
        return f"<JobQueueItem(id={self.id}, status='{self.status}')>"


class CommandEvent(Base):
    """Append-only log of every REPL command dispatch.

    Powers the intelligence layer: pattern detection, proactive suggestions,
    and named workflow capture. Written after every successful dispatch_command()
    call — one row, sub-millisecond, always-on.
    """

    __tablename__ = "command_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    command: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    args_json: Mapped[str] = mapped_column(Text, default="[]")       # JSON list of args
    context_file_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    context_tx_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), index=True
    )

    def __repr__(self) -> str:
        return f"<CommandEvent(id={self.id}, cmd='{self.command}', ts={self.created_at})>"


class Workflow(Base):
    """A named, replayable sequence of REPL commands.

    Created via \\workflow save <name>. Replayed via \\workflow run <name>.
    The steps field is a JSON array of {"command": str, "args": list[str]} objects.
    """

    __tablename__ = "workflows"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    steps_json: Mapped[str] = mapped_column(Text, default="[]")      # JSON list of steps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    def __repr__(self) -> str:
        return f"<Workflow(id={self.id}, name='{self.name}')>"


class NoteRecord(Base):
    """A user-authored note — the intake pipe for direct understanding into the namespace.
    
    The note is the only place where the user's understanding enters the expression
    namespace in their own voice, unmediated by AI or transcription. Everything else
    in the system receives the world and gives back understanding. The note is where
    the user gives the system their understanding directly.
    
    Lifecycle:
        draft    → ExpressionRecord not yet created. Not searchable.
        active   → ExpressionRecord exists and is embedded. Searchable via .search.
        archived → Hidden from default listings but preserved in the namespace.
    """
    __tablename__ = "notes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False, default="Untitled Note")
    body: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(16), default="draft")  # draft | active | archived
    expression_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("expressions.id", ondelete="SET NULL"), nullable=True, index=True
    )
    audio_file_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("audio_files.id", ondelete="SET NULL"), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )

    # Relationships
    audio_file: Mapped[AudioFileRecord | None] = relationship()
    expression: Mapped[ExpressionRecord | None] = relationship()

    def __repr__(self) -> str:
        return f"<NoteRecord(id={self.id}, title='{self.title}', status='{self.status}')>"
