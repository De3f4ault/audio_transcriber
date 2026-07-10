import logging
import json
from pydantic import ValidationError
from sqlalchemy import text as sql_text

from audiobench.core.db_session import get_session
from .operator_templates import _TEMPLATE_CLASSES, DaemonProposal
from audiobench.daemon.intelligence.calibration import get_calibration_tracker

logger = logging.getLogger("audiobench.daemon.intelligence.operator_registry")

class OperatorRegistry:
    def __init__(self):
        self.dynamic_operators = {}

    def load_from_db(self, store=None) -> None:
        logger.info("Loading dynamic operators from confirmed daemon_proposal records...")
        with get_session() as session:
            rows = session.execute(
                sql_text("""
                SELECT id, content FROM expressions
                WHERE source_type = 'daemon_proposal'
                AND inference_status = 'confirmed'
                """)
            ).fetchall()
            
            for row in rows:
                try:
                    data = json.loads(row[1])
                    self._register_proposal(row[0], data, session)
                except Exception as e:
                    logger.error(f"Failed to load operator {row[0]}: {e}")

    def hot_register(self, expression_id: int) -> None:
        logger.info("Hot-registering dynamic operator from proposal %d", expression_id)
        with get_session() as session:
            row = session.execute(
                sql_text("""
                SELECT content FROM expressions
                WHERE id = :eid AND source_type = 'daemon_proposal'
                """),
                {"eid": expression_id}
            ).fetchone()
            
            if row:
                try:
                    data = json.loads(row[0])
                    self._register_proposal(expression_id, data, session)
                except Exception as e:
                    logger.error(f"Failed to hot register operator {expression_id}: {e}")

    def _register_proposal(self, expression_id: int, proposal_data: dict, session) -> None:
        template_name = proposal_data.get("operator_template")
        schema_version = proposal_data.get("schema_version")
        parameters = proposal_data.get("parameters", {})
        
        if template_name not in _TEMPLATE_CLASSES:
            logger.warning(
                f"rejected: unknown operator_template '{template_name}' for proposal "
                f"{expression_id}. Template may have been removed or renamed."
            )
            session.execute(
                sql_text("UPDATE expressions SET inference_status = 'rejected' WHERE id = :eid"),
                {"eid": expression_id}
            )
            session.commit()
            return
            
        template_class = _TEMPLATE_CLASSES[template_name]
        
        if schema_version != template_class.SCHEMA_VERSION:
            logger.warning(f"Schema mismatch for {template_name}. Expected {template_class.SCHEMA_VERSION}, got {schema_version}")
            return
            
        try:
            instance = template_class(**parameters)
            self.dynamic_operators[expression_id] = instance
        except ValidationError as e:
            logger.error(f"Validation failed for proposal {expression_id}: {e}")
            session.execute(
                sql_text("UPDATE expressions SET inference_status = 'rejected' WHERE id = :eid"),
                {"eid": expression_id}
            )
            session.commit()

    def authorize(self, expression_id: int) -> None:
        """Mark a proposal confirmed in DB then hot-register it as a live operator.

        This is the single call the server handler makes. Status write and
        operator instantiation are kept together so the registry is never in
        a state where confirmed exists in DB but the operator is not resident.
        """
        with get_session() as session:
            session.execute(
                sql_text("UPDATE expressions SET inference_status = 'confirmed' WHERE id = :eid"),
                {"eid": expression_id}
            )
            session.commit()
        self.hot_register(expression_id)

    def deactivate(self, expression_id: int) -> None:
        if expression_id in self.dynamic_operators:
            del self.dynamic_operators[expression_id]
        with get_session() as session:
            session.execute(
                sql_text("UPDATE expressions SET inference_status = 'revoked' WHERE id = :eid"),
                {"eid": expression_id}
            )
            session.commit()

    def get_pending_proposals(self) -> list[dict]:
        logger.info("Fetching pending proposals and running staleness check...")
        valid_proposals = []
        with get_session() as session:
            rows = session.execute(
                sql_text("""
                SELECT id, content, created_at FROM expressions
                WHERE source_type = 'daemon_proposal'
                AND inference_status IN ('proposed', 'deferred')
                """)
            ).fetchall()
            
            tracker = get_calibration_tracker()
            
            for row in rows:
                try:
                    data = json.loads(row[1])
                    region_id = data.get("region_id", "")
                    
                    stats = tracker.get_region_stats(region_id)
                    new_samples = stats.total_since(row[2].timestamp() if hasattr(row[2], 'timestamp') else row[2])
                    
                    # Staleness check
                    if stats.confirm_rate < 0.70 or new_samples < 5:
                        logger.info(f"Proposal {row[0]} is stale. Rate: {stats.confirm_rate}, New samples: {new_samples}")
                        session.execute(
                            sql_text("UPDATE expressions SET inference_status = 'expired' WHERE id = :eid"),
                            {"eid": row[0]}
                        )
                        continue
                        
                    valid_proposals.append({
                        "id": row[0],
                        "data": data,
                        "created_at": row[2]
                    })
                except Exception as e:
                    logger.error(f"Failed to process proposal {row[0]}: {e}")
                    
            session.commit()
            
        return valid_proposals

_registry = None
def get_operator_registry() -> OperatorRegistry:
    global _registry
    if _registry is None:
        _registry = OperatorRegistry()
    return _registry
