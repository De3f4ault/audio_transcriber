import logging
import time
import json
from sqlalchemy import text as sql_text

from .scheduler import IntelligenceTask
from audiobench.daemon.server import _get_store
from audiobench.core.db_session import get_session
from audiobench.daemon.intelligence.calibration import get_calibration_tracker

logger = logging.getLogger("audiobench.daemon.intelligence.proposal_generator")

class ProposalGenerator(IntelligenceTask):
    INTERVAL_SECONDS = 7 * 86400  # 1 week

    async def run(self) -> None:
        logger.info("Running ProposalGenerator...")
        store = _get_store()
        tracker = get_calibration_tracker()
        now = time.time()
        
        with get_session() as session:
            # Check days of data
            earliest_ts = session.execute(
                sql_text("""
                SELECT MIN(created_at) FROM expressions
                WHERE source_type = 'daemon_calibration'
                """)
            ).scalar()
            
            if earliest_ts is None:
                return

            # SQLite stores created_at as an ISO-8601 string; convert to a Unix
            # timestamp before arithmetic so we don't crash with float - str.
            try:
                import datetime as _dt
                if isinstance(earliest_ts, str):
                    # Strip trailing Z / offset so fromisoformat works on Python 3.10-
                    ts_clean = earliest_ts.replace("Z", "+00:00")
                    earliest_dt = _dt.datetime.fromisoformat(ts_clean)
                    earliest_unix = earliest_dt.timestamp()
                else:
                    earliest_unix = float(earliest_ts)
            except Exception as parse_exc:
                logger.warning("ProposalGenerator: could not parse earliest_ts %r: %s", earliest_ts, parse_exc)
                return

            days_of_data = int((now - earliest_unix) / 86400)
            if days_of_data < 30:
                logger.info("Not enough calibration data (%d days). Needs >= 30.", days_of_data)
                return
            
            # Note: Hardcoded to one operator template for Phase 5 tests/stubs
            operator_template = "LectureModeOperator"
            schema_version = 1
            parameters_json = json.dumps({"frequency_multiplier": 1.5})
            
            for region_id, stats in tracker.stats.items():
                if stats.confirm_rate >= 0.70 and stats.total_inferences >= 20:
                    # Check for existing proposal
                    existing = session.execute(
                        sql_text("""
                        SELECT id FROM expressions
                        WHERE source_type = 'daemon_proposal'
                        AND inference_status IN ('proposed', 'deferred')
                        AND content LIKE :region_like
                        AND content LIKE :op_like
                        LIMIT 1
                        """),
                        {
                            "region_like": f"%region {region_id}%",
                            "op_like": f"%Activate {operator_template}%"
                        }
                    ).fetchone()
                    
                    if existing:
                        continue
                    
                    content = (
                        f"Proposal: Activate {operator_template} for region {region_id}.\n"
                        f"Evidence: confirm_rate={stats.confirm_rate:.2f} across {stats.total_inferences} inferences over {days_of_data} days.\n"
                        f"Proposed parameters: {parameters_json}.\n"
                        f"Schema version: {schema_version}."
                    )
                    
                    try:
                        store.add_expression(
                            source_type="daemon_proposal",
                            content=content,
                            inference_status="proposed"
                        )
                        logger.info(f"Generated proposal for region {region_id}")
                    except Exception as e:
                        logger.error(f"Failed to write proposal for {region_id}: {e}")
