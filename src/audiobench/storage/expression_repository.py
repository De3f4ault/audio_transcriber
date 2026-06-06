"""Repository — CRUD operations for Expression memory data.

Provides a clean interface for managing semantic memory expressions:
- Registering new expressions
- Linking expressions via directed relations
- Querying by ID and retrieving relations
- Navigating expression hierarchies (walking to parents)
"""

from __future__ import annotations

from typing import Literal

from sqlalchemy import asc

from audiobench.core.db_session import get_session
from audiobench.core.logger_factory import get_logger
from audiobench.memory.enums import RelationType
from audiobench.storage.models import ExpressionRecord, ExpressionRelation

logger = get_logger("storage.expression_repository")


class ExpressionRepository:
    """CRUD and traversal operations for semantic memory expressions."""

    def register(
        self,
        content: str,
        source_type: str,
        *,
        source_id: int | None = None,
        session_type: str | None = None,
        session_id: int | None = None,
        speaker: str | None = None,
    ) -> ExpressionRecord:
        """Register a new semantic expression.

        Returns:
            The created ExpressionRecord.
        """
        import hashlib

        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        with get_session() as session:
            # Check for existing semantic content within this source_type to prevent duplication
            existing = session.query(ExpressionRecord).filter_by(
                content_hash=content_hash,
                source_type=source_type
            ).first()
            if existing:
                logger.debug("Deduplicated expression #%d via content_hash", existing.id)
                session.expunge(existing)
                return existing

            record = ExpressionRecord(
                content=content,
                content_hash=content_hash,
                source_type=source_type,
                source_id=source_id,
                session_type=session_type,
                session_id=session_id,
                speaker=speaker,
            )
            session.add(record)
            session.commit()

            self._resolve_pending_relations(session, record)

            session.refresh(record)
            logger.info("Registered %s expression #%d", source_type, record.id)
            # Create a detached instance to return outside session
            session.expunge(record)
            return record

    def _resolve_pending_relations(self, session, record: ExpressionRecord) -> None:
        """Resolve any pending forward references to this newly created expression."""
        if record.source_id is None:
            return

        from audiobench.storage.note_repository import PendingRelation
        pending = session.query(PendingRelation).filter_by(
            to_expression_id_hint=record.source_id,
            to_source_type=record.source_type
        ).all()

        if pending:
            for p in pending:
                rel = ExpressionRelation(
                    from_expression_id=p.from_expression_id,
                    to_expression_id=record.id,
                    relation_type=p.relation_type,
                    created_by="system"
                )
                session.add(rel)
                session.delete(p)
            session.commit()
            logger.info("Resolved %d pending relations for expression #%d", len(pending), record.id)

    def get_by_id(self, expression_id: int) -> ExpressionRecord | None:
        """Get a single expression by ID."""
        with get_session() as session:
            record = session.query(ExpressionRecord).filter_by(id=expression_id).first()
            if record:
                session.expunge(record)
            return record

    def link(
        self,
        from_id: int,
        to_id: int,
        relation_type: str,
        *,
        weight: float = 1.0,
        created_by: str = "system",
    ) -> ExpressionRelation:
        """Create a directed relation between two expressions.

        Returns:
            The created ExpressionRelation.
        """
        with get_session() as session:
            relation = ExpressionRelation(
                from_expression_id=from_id,
                to_expression_id=to_id,
                relation_type=relation_type,
                weight=weight,
                created_by=created_by,
            )
            session.add(relation)
            session.commit()
            session.refresh(relation)
            logger.info("Linked %d -> %d (type: %s)", from_id, to_id, relation_type)
            session.expunge(relation)
            return relation

    def get_relations(
        self,
        expression_id: int,
        direction: Literal["out", "in", "both"] = "both",
        relation_type: str | None = None,
    ) -> list[ExpressionRelation]:
        """Get relations involving this expression.

        Args:
            expression_id: The ID of the expression.
            direction: 'out' (where this is source), 'in' (where this is target), or 'both'.
            relation_type: Optional filter by relation type.

        Returns:
            List of ExpressionRelation records.
        """
        with get_session() as session:
            query = session.query(ExpressionRelation)

            if direction == "out":
                query = query.filter_by(from_expression_id=expression_id)
            elif direction == "in":
                query = query.filter_by(to_expression_id=expression_id)
            else:
                query = query.filter(
                    (ExpressionRelation.from_expression_id == expression_id)
                    | (ExpressionRelation.to_expression_id == expression_id)
                )

            if relation_type:
                query = query.filter_by(relation_type=relation_type)

            records = query.order_by(asc(ExpressionRelation.created_at)).all()
            for r in records:
                session.expunge(r)
            return records

    def update_inference_status(
        self,
        expression_id: int,
        status: str,
    ) -> bool:
        """Update the inference status of an expression.

        Args:
            expression_id: The ID of the expression.
            status: Must match InferenceStatus values.

        Returns:
            True if updated, False if not found.
        """
        with get_session() as session:
            record = session.query(ExpressionRecord).filter_by(id=expression_id).first()
            if not record:
                return False

            record.inference_status = status
            session.commit()
            logger.info("Updated expression #%d inference_status to '%s'", expression_id, status)
            return True

    def walk_to_parent(self, expression_id: int) -> ExpressionRecord | None:
        """Follow a 'source' relation upward to find the parent expression.

        Returns the target expression of the first outbound 'source' relation found,
        or None if no such relation exists.
        """
        out_relations = self.get_relations(
            expression_id, direction="out", relation_type=RelationType.SOURCE.value
        )

        if not out_relations:
            return None

        # Follow the first source relation
        parent_id = out_relations[0].to_expression_id
        return self.get_by_id(parent_id)
