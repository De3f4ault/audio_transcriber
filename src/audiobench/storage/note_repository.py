"""Repository — CRUD operations and lifecycle management for Note data.

Handles the creation, updates, and parsing of user notes. When a note is saved,
its body is parsed for explicit references ([[...]]), and an ExpressionRecord is
created or updated, automatically queuing it for semantic embedding.
"""

from __future__ import annotations

import re
from datetime import datetime, UTC

from audiobench.core.db_session import get_session
from audiobench.core.logger_factory import get_logger
from audiobench.memory.enums import SourceType, RelationType
from audiobench.storage.models import (
    NoteRecord,
    ExpressionRecord,
    ExpressionRelation,
    AudioFileRecord,
    TranscriptionRecord,
    SegmentRecord,
    ChatMessage,
    ChatConversation,
    AskEntry,
    BookmarkRecord
)
from audiobench.storage.models import Base  # To reflect the PendingRelation we will define

# We need to map the PendingRelation here, although it was added via migration
from sqlalchemy import Integer, String, Text, ForeignKey, DateTime, Column
from sqlalchemy.orm import Mapped, mapped_column

logger = get_logger("storage.note_repository")

# Reference parsing regex: matches [[...]]
REF_PATTERN = re.compile(r'\[\[([^\]]+)\]\]')


class PendingRelation(Base):
    __tablename__ = "pending_relations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    from_expression_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("expressions.id", ondelete="CASCADE"), nullable=False
    )
    to_expression_id_hint: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    to_source_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    relation_type: Mapped[str] = mapped_column(String(64), nullable=False, default="explicit")
    raw_ref: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))


class NoteRepository:
    """CRUD operations for NoteRecord."""

    def create(self, title: str, audio_file_id: int | None = None) -> NoteRecord:
        with get_session() as session:
            note = NoteRecord(
                title=title,
                audio_file_id=audio_file_id,
                status="draft",
            )
            session.add(note)
            session.commit()
            session.refresh(note)
            session.expunge(note)
            return note

    def get_by_id(self, note_id: int) -> NoteRecord | None:
        with get_session() as session:
            note = session.query(NoteRecord).filter_by(id=note_id).first()
            if note:
                session.expunge(note)
            return note

    def list_notes(self, status: str = "active", limit: int = 50) -> list[NoteRecord]:
        with get_session() as session:
            notes = session.query(NoteRecord).filter_by(status=status).order_by(NoteRecord.updated_at.desc()).limit(limit).all()
            for n in notes:
                session.expunge(n)
            return notes

    def save_body(self, note_id: int, body: str) -> NoteRecord:
        """Update body, change status to active, write ExpressionRecord, queue embed."""
        import hashlib
        from audiobench.daemon.client import DaemonClient

        with get_session() as session:
            note = session.query(NoteRecord).filter_by(id=note_id).first()
            if not note:
                raise ValueError(f"Note {note_id} not found.")

            note.body = body
            
            # Content hash for dedup/updates
            content_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()

            # Handle ExpressionRecord
            expr = None
            if note.status == "draft" or note.expression_id is None:
                expr = ExpressionRecord(
                    content=body,
                    content_hash=content_hash,
                    source_type=SourceType.JOURNAL_ENTRY.value,
                    source_id=note.id,
                    inference_status="active"
                )
                session.add(expr)
                session.flush() # get expr.id
                note.expression_id = expr.id
                note.status = "active"
            else:
                expr = session.query(ExpressionRecord).filter_by(id=note.expression_id).first()
                if expr:
                    expr.content = body
                    expr.content_hash = content_hash
            
            # Save parsing for explicit relations
            if expr:
                self._parse_and_link_references(session, expr.id, body)

            session.commit()
            session.refresh(note)
            
            # Queue semantic embedding
            if expr:
                daemon = DaemonClient()
                if daemon.is_alive():
                    daemon.embed(expr.id)

            session.expunge(note)
            return note

    def _parse_and_link_references(self, session, note_expr_id: int, body: str) -> None:
        """Parse [[...]] syntax and create explicit ExpressionRelations or PendingRelations."""
        
        # Clear existing explicit relations stemming from this note
        session.query(ExpressionRelation).filter_by(
            from_expression_id=note_expr_id,
            relation_type=RelationType.EXPLICIT.value,
            created_by="user"
        ).delete()
        
        session.query(PendingRelation).filter_by(
            from_expression_id=note_expr_id
        ).delete()

        matches = REF_PATTERN.findall(body)
        if not matches:
            return

        for ref in matches:
            ref = ref.strip().lower()
            target_expr_id = None
            to_source_type_hint = None
            to_id_hint = None

            try:
                if ref.startswith("#") or ref.startswith("transcript:"):
                    # [[#42]] or [[transcript:42]]
                    raw_id = ref.replace("#", "").replace("transcript:", "").strip()
                    # Handle "42 at 14:32"
                    if " at " in raw_id:
                        tx_id_str, time_str = raw_id.split(" at ", 1)
                        tx_id = int(tx_id_str.strip())
                        # In a full implementation, we'd find the closest segment here
                        # For now, we fallback to the transcript expression
                        target_expr = session.query(ExpressionRecord).filter_by(
                            source_id=tx_id, source_type=SourceType.AUDIO_TRANSCRIPT.value
                        ).first()
                        if target_expr:
                            target_expr_id = target_expr.id
                    else:
                        tx_id = int(raw_id)
                        target_expr = session.query(ExpressionRecord).filter_by(
                            source_id=tx_id, source_type=SourceType.AUDIO_TRANSCRIPT.value
                        ).first()
                        if target_expr:
                            target_expr_id = target_expr.id
                        else:
                            # Might be pending
                            to_id_hint = tx_id
                            to_source_type_hint = SourceType.AUDIO_TRANSCRIPT.value

                elif ref.startswith("note:"):
                    n_id = int(ref.replace("note:", "").strip())
                    note = session.query(NoteRecord).filter_by(id=n_id).first()
                    if note and note.expression_id:
                        target_expr_id = note.expression_id
                    else:
                        to_id_hint = n_id
                        to_source_type_hint = SourceType.JOURNAL_ENTRY.value

                elif ref.startswith("ask_answer:"):
                    e_id = int(ref.replace("ask_answer:", "").strip())
                    expr = session.query(ExpressionRecord).filter_by(id=e_id, source_type=SourceType.ASK_ANSWER.value).first()
                    if expr:
                        target_expr_id = expr.id
                        
                elif ref.startswith("chat:"):
                    # [[chat:42 turn:7]] - simplify to just linking chat conversation
                    if " turn:" in ref:
                        c_id_str, turn_str = ref.replace("chat:", "").split(" turn:")
                        c_id = int(c_id_str.strip())
                        expr = session.query(ExpressionRecord).filter_by(session_id=c_id, source_type=SourceType.CHAT_MESSAGE.value).first() # approximation
                        if expr:
                            target_expr_id = expr.id
                            
                elif ref.startswith("bookmark:"):
                    b_id = int(ref.replace("bookmark:", "").strip())
                    expr = session.query(ExpressionRecord).filter_by(source_id=b_id, source_type=SourceType.BOOKMARK_NOTE.value).first()
                    if expr:
                        target_expr_id = expr.id

                elif ref.startswith("expression:"):
                    e_id = int(ref.replace("expression:", "").strip())
                    expr = session.query(ExpressionRecord).filter_by(id=e_id).first()
                    if expr:
                        target_expr_id = expr.id
            except ValueError:
                # Could not parse integer IDs
                continue
                
            if target_expr_id:
                # Create real relation
                rel = ExpressionRelation(
                    from_expression_id=note_expr_id,
                    to_expression_id=target_expr_id,
                    relation_type=RelationType.EXPLICIT.value,
                    created_by="user"
                )
                session.add(rel)
            elif to_id_hint:
                # Create pending relation
                prel = PendingRelation(
                    from_expression_id=note_expr_id,
                    to_expression_id_hint=to_id_hint,
                    to_source_type=to_source_type_hint,
                    relation_type=RelationType.EXPLICIT.value,
                    raw_ref=ref
                )
                session.add(prel)

    def find_or_create_context_note(self, audio_file_id: int, label: str) -> NoteRecord:
        with get_session() as session:
            title = f"Notes on {label}"
            note = session.query(NoteRecord).filter_by(audio_file_id=audio_file_id, title=title).first()
            if not note:
                note = NoteRecord(
                    title=title,
                    audio_file_id=audio_file_id,
                    status="draft"
                )
                session.add(note)
                session.commit()
                session.refresh(note)
            session.expunge(note)
            return note

    def find_or_create_inbox(self) -> NoteRecord:
        with get_session() as session:
            note = session.query(NoteRecord).filter_by(title="Inbox").first()
            if not note:
                note = NoteRecord(
                    title="Inbox",
                    status="draft"
                )
                session.add(note)
                session.commit()
                session.refresh(note)
            session.expunge(note)
            return note

    def append_capture(self, note_id: int, text: str, expression_id: int | None = None) -> None:
        with get_session() as session:
            note = session.query(NoteRecord).filter_by(id=note_id).first()
            if not note:
                return

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            entry = f"[{timestamp}] {text}\n"
            
            if note.body:
                note.body += f"\n{entry}"
            else:
                note.body = entry
                
            session.commit()
            
        # Re-save body to trigger embedding and relation parsing
        self.save_body(note_id, note.body)

        # For structured captures (expression_id provided), we add relation directly if it doesn't exist
        if expression_id and note.expression_id:
            with get_session() as session:
                existing = session.query(ExpressionRelation).filter_by(
                    from_expression_id=note.expression_id,
                    to_expression_id=expression_id,
                    relation_type=RelationType.EXPLICIT.value
                ).first()
                if not existing:
                    rel = ExpressionRelation(
                        from_expression_id=note.expression_id,
                        to_expression_id=expression_id,
                        relation_type=RelationType.EXPLICIT.value,
                        created_by="user"
                    )
                    session.add(rel)
                    session.commit()

    def list_unprocessed_captures(self) -> list[dict]:
        # This function is meant to show unprocessed captures. For now, we will return 
        # all lines that look like captures from all notes. In a real system, we'd add
        # a way to mark captures as processed, perhaps a separate table or a specific syntax.
        # Given the requirements, we'll extract them from the bodies.
        captures = []
        pattern = re.compile(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] (.*)')
        
        with get_session() as session:
            notes = session.query(NoteRecord).filter(NoteRecord.body.isnot(None)).all()
            for note in notes:
                for line in note.body.split('\n'):
                    match = pattern.match(line)
                    if match:
                        captures.append({
                            "timestamp": match.group(1),
                            "text": match.group(2).strip(),
                            "note_id": note.id,
                            "note_title": note.title
                        })
        
        # Sort reverse chronological
        captures.sort(key=lambda x: x["timestamp"], reverse=True)
        return captures
