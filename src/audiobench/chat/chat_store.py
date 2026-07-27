"""Repository — CRUD operations for AI chat conversations.

Provides persistence for chat sessions:
- Creating and updating conversations
- Saving messages with role/content/thinking
- Listing and resuming past conversations
- Deleting conversations
"""

from __future__ import annotations

import json

from sqlalchemy import desc

from audiobench.core.db_session import get_session
from audiobench.core.logger_factory import get_logger
from audiobench.storage.models import ChatConversation, ChatMessage

logger = get_logger("storage.chat_repository")


class ChatRepository:
    """CRUD operations for AI chat persistence."""

    def create_conversation(
        self,
        model: str,
        transcript_ids: list[int] | None = None,
        title: str = "Untitled Chat",
        session_type: str = "chat",
    ) -> int:
        """Create a new chat conversation.

        Args:
            model: The Ollama model name.
            transcript_ids: List of transcript IDs loaded as context.
            title: Conversation title (will be AI-generated later).
            session_type: Type of session (chat, search_followup, etc).

        Returns:
            The conversation ID.
        """
        with get_session() as session:
            conv = ChatConversation(
                title=title,
                model_name=model,
                transcript_ids=json.dumps(transcript_ids or []),
                session_type=session_type,
                message_count=0,
            )
            session.add(conv)
            session.commit()
            logger.info("Created conversation #%d (model=%s)", conv.id, model)
            return conv.id

    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        thinking: str | None = None,
        model_name: str | None = None,
    ) -> int:
        """Add a message to a conversation.

        Args:
            conversation_id: The conversation to add to.
            role: Message role — "system", "user", or "assistant".
            content: The message text.
            thinking: Optional chain-of-thought text.
            model_name: Optional model name (for comparison pairs).

        Returns:
            The message ID.
        """
        with get_session() as session:
            msg = ChatMessage(
                conversation_id=conversation_id,
                role=role,
                content=content,
                thinking=thinking,
                model_name=model_name,
                token_count=len(content) // 4,  # rough estimate
            )
            session.add(msg)

            # Update conversation message count
            conv = session.query(ChatConversation).filter_by(id=conversation_id).first()
            if conv:
                conv.message_count = session.query(ChatMessage).filter_by(
                    conversation_id=conversation_id
                ).filter(ChatMessage.role != "system").count() + (1 if role != "system" else 0)

            session.commit()

            # --- PHASE 5: Expression Registration ---
            if role != "system":
                try:
                    from audiobench.memory.knowledge_ingester import KnowledgeIngester
                    import threading
                    
                    ingester = KnowledgeIngester()
                    
                    # session.expunge is needed to detach the objects so they can be accessed safely from the thread
                    session.expunge(msg)
                    if conv:
                        session.expunge(conv)
                    
                    # Run ingestion in a background thread to not block the chat
                    threading.Thread(
                        target=ingester.ingest_chat_message,
                        args=(msg, conv),
                        daemon=True
                    ).start()
                except Exception as e:
                    logger.error("Failed to spawn ingestion thread for chat message %d: %s", msg.id, e)

            return msg.id

    def get_conversation(self, conversation_id: int) -> dict | None:
        """Get a conversation with all its messages.

        Returns:
            Dict with conversation metadata and messages, or None.
        """
        with get_session() as session:
            conv = session.query(ChatConversation).filter_by(id=conversation_id).first()
            if conv is None:
                return None

            return {
                "id": conv.id,
                "title": conv.title,
                "model": conv.model_name,
                "transcript_ids": json.loads(conv.transcript_ids),
                "message_count": conv.message_count,
                "created_at": (conv.created_at.isoformat() if conv.created_at else ""),
                "updated_at": (conv.updated_at.isoformat() if conv.updated_at else ""),
                "messages": [
                    {
                        "id": msg.id,
                        "role": msg.role,
                        "content": msg.content,
                        "thinking": msg.thinking,
                        "model_name": msg.model_name,
                        "token_count": msg.token_count,
                        "created_at": (msg.created_at.isoformat() if msg.created_at else ""),
                    }
                    for msg in sorted(conv.messages, key=lambda m: m.created_at)
                ],
            }

    def get_messages_for_api(self, conversation_id: int) -> list[dict]:
        """Get messages formatted for the Ollama /api/chat endpoint.

        Returns:
            List of {"role": str, "content": str} dicts.
        """
        with get_session() as session:
            messages = (
                session.query(ChatMessage)
                .filter_by(conversation_id=conversation_id)
                .order_by(ChatMessage.created_at)
                .all()
            )
            return [{"role": msg.role, "content": msg.content} for msg in messages]

    def list_conversations(self, limit: int = 20) -> list[dict]:
        """List recent conversations (summary view).

        Returns:
            List of conversation summary dicts.
        """
        with get_session() as session:
            convs = (
                session.query(ChatConversation)
                .order_by(desc(ChatConversation.updated_at))
                .limit(limit)
                .all()
            )
            return [
                {
                    "id": c.id,
                    "title": c.title,
                    "model": c.model_name,
                    "message_count": c.message_count,
                    "transcript_ids": json.loads(c.transcript_ids),
                    "created_at": (c.created_at.isoformat() if c.created_at else ""),
                    "updated_at": (c.updated_at.isoformat() if c.updated_at else ""),
                }
                for c in convs
            ]

    def update_title(self, conversation_id: int, title: str) -> None:
        """Update a conversation's title."""
        with get_session() as session:
            conv = session.query(ChatConversation).filter_by(id=conversation_id).first()
            if conv:
                conv.title = title[:256]
                session.commit()
                logger.info(
                    "Updated title for conversation #%d: %s",
                    conversation_id,
                    title[:50],
                )

    def update_transcript_ids(self, conversation_id: int, transcript_ids: list[int]) -> None:
        """Update the transcript IDs associated with a conversation."""
        with get_session() as session:
            conv = session.query(ChatConversation).filter_by(id=conversation_id).first()
            if conv:
                conv.transcript_ids = json.dumps(transcript_ids)
                session.commit()

    def delete_conversation(self, conversation_id: int) -> bool:
        """Delete a conversation. Returns True if deleted."""
        with get_session() as session:
            conv = session.query(ChatConversation).filter_by(id=conversation_id).first()
            if not conv:
                return False

            from audiobench.storage.expression_repository import ExpressionRepository
            from audiobench.memory.enums import SourceType

            ExpressionRepository().delete_by_source(
                SourceType.CHAT.value, conversation_id
            )
            session.delete(conv)
            session.commit()
            logger.info("Deleted conversation #%d", conversation_id)
            return True

    # ------------------------------------------------------------------
    # Ask Log Operations
    # ------------------------------------------------------------------

    def get_or_create_ask_log(self, audio_file_id: int) -> int:
        """Get the AskLog ID for an audio file, creating it if necessary."""
        from audiobench.storage.models import AskLog

        with get_session() as session:
            log = session.query(AskLog).filter_by(audio_file_id=audio_file_id).first()
            if log:
                return log.id

            log = AskLog(audio_file_id=audio_file_id, entry_count=0)
            session.add(log)
            session.commit()
            return log.id

    def add_ask_entry(
        self,
        log_id: int,
        question: str,
        answer: str,
        model_name: str,
        question_expression_id: int | None = None,
        answer_expression_id: int | None = None,
    ) -> int:
        """Add an entry to an AskLog."""
        from audiobench.storage.models import AskEntry, AskLog

        with get_session() as session:
            entry = AskEntry(
                log_id=log_id,
                question=question,
                answer=answer,
                model_name=model_name,
                token_count=(len(question) + len(answer)) // 4,
                question_expression_id=question_expression_id,
                answer_expression_id=answer_expression_id,
            )
            session.add(entry)

            log = session.query(AskLog).filter_by(id=log_id).first()
            if log:
                log.entry_count = session.query(AskEntry).filter_by(log_id=log_id).count() + 1

            session.commit()
            return entry.id

    def delete_all_conversations(self) -> int:
        """Delete all conversations. Returns number deleted."""
        with get_session() as session:
            conv_ids = [r[0] for r in session.query(ChatConversation.id).all()]
            from audiobench.storage.expression_repository import ExpressionRepository
            from audiobench.memory.enums import SourceType

            expr_repo = ExpressionRepository()
            for cid in conv_ids:
                expr_repo.delete_by_source(SourceType.CHAT.value, cid)

            count = len(conv_ids)
            session.query(ChatMessage).delete()
            session.query(ChatConversation).delete()
            session.commit()
            logger.info("Deleted %d conversation(s)", count)
            return count

    # ------------------------------------------------------------------
    # Summary Operations
    # ------------------------------------------------------------------

    def save_summary(
        self,
        conversation_id: int,
        narrative: str,
        drift_phases: str,
        key_insights: str,
        open_threads: str,
        refined_title: str | None,
        generated_by: str,
        expression_id: int | None = None,
    ) -> int:
        """Save a ConversationSummary."""
        from audiobench.storage.models import ConversationSummary

        with get_session() as session:
            summary = (
                session.query(ConversationSummary)
                .filter_by(conversation_id=conversation_id)
                .first()
            )
            if not summary:
                summary = ConversationSummary(conversation_id=conversation_id)
                session.add(summary)

            summary.narrative = narrative
            summary.drift_phases = drift_phases
            summary.key_insights = key_insights
            summary.open_threads = open_threads
            if refined_title:
                summary.refined_title = refined_title
            summary.generated_by = generated_by
            if expression_id is not None:
                summary.expression_id = expression_id

            session.commit()
            return summary.id

    def update_summary_expression(self, summary_id: int, expression_id: int) -> None:
        """Update the expression ID for a ConversationSummary."""
        from audiobench.storage.models import ConversationSummary

        with get_session() as session:
            summary = session.query(ConversationSummary).filter_by(id=summary_id).first()
            if summary:
                summary.expression_id = expression_id
                session.commit()
