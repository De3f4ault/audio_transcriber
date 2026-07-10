import logging
import json
import numpy as np
from sqlalchemy import text as sql_text

from .scheduler import IntelligenceTask
from .math_utils import pairwise_cosine_above_threshold
from audiobench.core.db_session import get_session
from audiobench.daemon.server import _get_store
from audiobench.storage.models import PendingRelation

logger = logging.getLogger("audiobench.daemon.intelligence.connection_surfer")

class ConnectionSurfer(IntelligenceTask):
    INTERVAL_SECONDS = 3600 * 6  # Runs less frequently
    SIM_THRESHOLD = 0.85
    MAX_PER_SESSION = 10
    SAMPLE_SIZE = 200

    async def run(self) -> None:
        logger.info("Running ConnectionSurfer...")
        store = _get_store()
        
        seen_pairs = set()
        with get_session() as session:
            row = session.execute(
                sql_text("""
                SELECT content FROM expressions
                WHERE source_type = 'daemon_calibration'
                ORDER BY created_at DESC LIMIT 1
                """)
            ).fetchone()
            
            if row and row[0]:
                try:
                    calib_data = json.loads(row[0])
                    for a, b in calib_data.get("surfer_seen_pairs", []):
                        seen_pairs.add((a, b))
                except Exception:
                    pass

        # Sample 200 expressions with their audio_file_id
        # We join expressions -> expression_segment_map -> segments to get audio_file_id
        with get_session() as session:
            rows = session.execute(
                sql_text("""
                SELECT e.id, tx.audio_file_id, e.content
                FROM expressions e
                JOIN transcriptions tx ON e.source_id = tx.id
                WHERE e.source_type = 'audio_transcript'
                ORDER BY e.created_at DESC
                LIMIT :limit
                """),
                {"limit": self.SAMPLE_SIZE}
            ).mappings().all()

        if len(rows) < 2:
            return

        expr_ids = [r["id"] for r in rows]
        vectors = store.get_vectors(expr_ids)
        
        valid_rows = [r for r in rows if r["id"] in vectors]
        if len(valid_rows) < 2:
            return
            
        vec_list = [vectors[r["id"]] for r in valid_rows]
        id_list = [r["id"] for r in valid_rows]
        
        pairs = pairwise_cosine_above_threshold(vec_list, id_list, self.SIM_THRESHOLD)
        
        # Valid rows by id mapping
        row_map = {r["id"]: r for r in valid_rows}
        
        new_seen = []
        added_count = 0
        
        with get_session() as session:
            for id_a, id_b, sim in pairs:
                row_a = row_map[id_a]
                row_b = row_map[id_b]
                
                # Check different files
                if row_a["audio_file_id"] == row_b["audio_file_id"]:
                    continue
                    
                id_a = row_a["id"]
                id_b = row_b["id"]
                
                pair_key = (min(id_a, id_b), max(id_a, id_b))
                if pair_key in seen_pairs:
                    continue
                    
                seen_pairs.add(pair_key)
                new_seen.append(pair_key)
                
                if added_count >= self.MAX_PER_SESSION:
                    continue
                
                # Calculate sim
                # sim is already calculated
                
                title_a = f"File {row_a['audio_file_id']}"
                title_b = f"File {row_b['audio_file_id']}"
                preview_a = row_a["content"][:100]
                preview_b = row_b["content"][:100]
                
                content = (
                    f"Potential connection detected between two recordings.\n"
                    f"Expression {id_a} (source: {title_a}) and expression {id_b} (source: {title_b})\n"
                    f"share a semantic similarity of {sim:.2f}.\n"
                    f"Content preview A: \"{preview_a}\"\n"
                    f"Content preview B: \"{preview_b}\""
                )
                
                store.add_expression(
                    source_type="potential_relation",
                    content=content,
                    inference_status="proposed"
                )
                
                pr = PendingRelation(
                    from_expression_id=id_a,
                    to_expression_id_hint=id_b,
                    relation_type="semantic_link"
                )
                session.add(pr)
                added_count += 1
                
            session.commit()
            
        if new_seen:
            calib = {"surfer_seen_pairs": list(seen_pairs)}
            store.add_expression(
                source_type="daemon_calibration",
                content=json.dumps(calib)
            )
