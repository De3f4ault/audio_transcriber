#!/usr/bin/env python3
"""Automated lifecycle verification test for vector and expression deletion.

Proves that deleting a transcription via TranscriptionRepository.delete_by_id()
completely removes its associated ExpressionRecord rows from SQLite and vector nodes from LanceDB.
"""

import logging
import sys

sys.path.insert(0, "src")

from audiobench.core.db_session import get_session
from audiobench.storage.repository import TranscriptionRepository
from audiobench.storage.expression_repository import ExpressionRepository
from audiobench.storage.models import ExpressionRecord, TranscriptionRecord
from audiobench.transcribe.transcription_result import AudioMetadata
from audiobench.memory.enums import SourceType
from audiobench.memory.memory_store import MemoryStore

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
logger = logging.getLogger("test_lifecycle")


def main():
    logger.info("=== STARTING VECTOR DELETION LIFECYCLE TEST ===")
    repo = TranscriptionRepository()
    expr_repo = ExpressionRepository()
    store = MemoryStore()

    # 1. Create a dummy transcription record
    meta = AudioMetadata(
        file_path="/tmp/test_lifecycle_dummy.wav",
        file_name="test_lifecycle_dummy.wav",
        file_hash="dummy_hash_lifecycle_12345",
        file_size=1024,
        duration=10.0,
        format="wav",
        sample_rate=16000,
        channels=1,
    )
    tx_id = repo.begin_transcription(audio_metadata=meta, engine="faster-whisper", model_name="tiny")
    logger.info("Created dummy transcription #%d", tx_id)

    # 2. Register 5 expressions for this transcription
    batch = [
        {
            "_hash": f"hash_lifecycle_test_{tx_id}_{i}",
            "content": f"This is test sentence number {i} for lifecycle deletion.",
            "source_type": SourceType.AUDIO_TRANSCRIPT.value,
            "source_id": tx_id,
        }
        for i in range(5)
    ]
    expr_records = expr_repo.register_batch(batch)
    expr_ids = [r.id for r in expr_records]
    logger.info("Registered %d expressions in SQLite: %s", len(expr_ids), expr_ids)

    assert len(expr_ids) == 5, f"Expected 5 expressions registered, got {len(expr_ids)}"

    # 3. Add 5 vector nodes to LanceDB using write_node to match exact ExpressionNode schema
    for eid, rec in zip(expr_ids, expr_records):
        store.write_node(
            expression_id=eid,
            content=rec.content,
            source_type=rec.source_type,
        )
    logger.info("Added 5 corresponding vector nodes to LanceDB via write_node.")

    # Verify before deletion
    with get_session() as session:
        sql_count_before = session.query(ExpressionRecord).filter(ExpressionRecord.id.in_(expr_ids)).count()
    
    ids_str = ", ".join(str(eid) for eid in expr_ids)
    lancedb_rows_before = store.table.search().where(f"expression_id IN ({ids_str})").to_list()
    lancedb_count_before = len(lancedb_rows_before)

    logger.info("PRE-DELETION CHECK -> SQLite rows: %d | LanceDB vectors: %d", sql_count_before, lancedb_count_before)
    assert sql_count_before == 5, f"Expected 5 SQLite rows before delete, found {sql_count_before}"
    assert lancedb_count_before == 5, f"Expected 5 LanceDB nodes before delete, found {lancedb_count_before}"

    # 4. Perform deletion via TranscriptionRepository.delete_by_id()
    logger.info("Calling TranscriptionRepository().delete_by_id(%d)...", tx_id)
    success = repo.delete_by_id(tx_id)
    assert success is True, "delete_by_id returned False!"

    # 5. Verify post-deletion zero leakage
    with get_session() as session:
        sql_count_after = session.query(ExpressionRecord).filter(ExpressionRecord.id.in_(expr_ids)).count()
    
    lancedb_rows_after = store.table.search().where(f"expression_id IN ({ids_str})").to_list()
    lancedb_count_after = len(lancedb_rows_after)

    logger.info("POST-DELETION CHECK -> SQLite rows: %d | LanceDB vectors: %d", sql_count_after, lancedb_count_after)
    assert sql_count_after == 0, f"LEAK DETECTED: Found {sql_count_after} SQLite rows remaining!"
    assert lancedb_count_after == 0, f"GHOST VECTOR LEAK DETECTED: Found {lancedb_count_after} LanceDB nodes remaining!"

    logger.info("=== SUCCESS: ALL ASSERTS PASSED! 0 GHOST VECTORS LEAKED ===")


if __name__ == "__main__":
    main()
