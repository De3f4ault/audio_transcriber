"""Inference writer — handles writing system inferences to memory."""

from typing import Any

from audiobench.core.logger_factory import get_logger
from audiobench.daemon.factory import get_daemon_client
from audiobench.memory.enums import RelationType, SourceType
from audiobench.storage.expression_repository import ExpressionRepository

logger = get_logger("memory.inference")


class InferenceWriter:
    """Writes system inferences into the memory graph."""

    def __init__(self):
        self.expr_repo = ExpressionRepository()
        self.daemon = get_daemon_client()

    def write_inference(
        self, content: str, confidence: float, source_expression_ids: list[int]
    ) -> Any:  # Returns ExpressionRecord
        """Write an inference and link it to its source expressions."""

        # 1. Write Expression
        inference_expr = self.expr_repo.register(
            content=content,
            source_type=SourceType.SYSTEM_INFERENCE.value,
        )

        # Update extra fields directly
        from audiobench.core.db_session import get_session
        from audiobench.storage.models import ExpressionRecord

        with get_session() as session:
            rec = session.query(ExpressionRecord).filter_by(id=inference_expr.id).first()
            if rec:
                rec.inference_confidence = confidence
                rec.inference_status = "active"
                session.commit()
                # update returned object
                inference_expr.inference_confidence = confidence
                inference_expr.inference_status = "active"

        # 2. Write Relations
        for src_id in source_expression_ids:
            self.expr_repo.link(
                from_id=inference_expr.id, to_id=src_id, relation_type=RelationType.THEMATIC.value
            )

        # 3. Daemon send infer command
        # Wait, the protocol might not have 'infer' if it wasn't added to DaemonClient.
        # But we can just use `embed` as instructed, or if `infer` is required by daemon:
        try:
            # Let's use daemon.send directly since it's a custom command
            self.daemon._send(
                "infer",
                {
                    "expression_id": inference_expr.id,
                    "content": content,
                    "source_ids": source_expression_ids,
                },
            )
        except Exception as e:
            logger.warning("Daemon infer command failed: %s", e)

        return inference_expr
