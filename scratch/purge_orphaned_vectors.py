#!/usr/bin/env python3
"""Defensive, batched, resumable purge utility for orphaned expressions and semantic vectors.

By default, runs in review/dry-run mode without modifying any data.
Run with --execute to perform batched deletions.
"""

import argparse
import logging
import sys
from collections import Counter

sys.path.insert(0, "src")

from audiobench.core.db_session import get_session
from audiobench.storage.models import (
    ExpressionRecord,
    TranscriptionRecord,
    ExpressionRelation,
    ChatConversation,
)
from audiobench.memory.memory_store import MemoryStore

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
logger = logging.getLogger("purge_orphans")


def get_orphan_sqlite_ids(session) -> list[int]:
    """Return all ExpressionRecord IDs that point to non-existent sources."""
    live_tx_ids = {r[0] for r in session.query(TranscriptionRecord.id).all()}
    live_chat_ids = {r[0] for r in session.query(ChatConversation.id).all()}

    orphans = []
    all_exprs = session.query(
        ExpressionRecord.id, ExpressionRecord.source_type, ExpressionRecord.source_id
    ).all()
    for eid, stype, sid in all_exprs:
        if stype == "audio_transcript":
            if sid not in live_tx_ids:
                orphans.append((eid, stype, sid))
        elif stype == "chat":
            if sid not in live_chat_ids:
                orphans.append((eid, stype, sid))
    return orphans


def get_orphan_lancedb_ids(store: MemoryStore, valid_sqlite_ids: set[int]) -> list[int]:
    """Return all LanceDB expression_ids not present in valid SQLite expressions."""
    rows = store.table.search().select(["expression_id"]).to_list()
    orphans = []
    for r in rows:
        eid = int(r["expression_id"])
        if eid not in valid_sqlite_ids:
            orphans.append(eid)
    return orphans


def run_review():
    logger.info("=== STARTING DRY-RUN / REVIEW MODE ===")
    with get_session() as session:
        orphans = get_orphan_sqlite_ids(session)
        valid_sqlite_ids = {
            r[0]
            for r in session.query(ExpressionRecord.id).all()
        } - {eid for eid, _, _ in orphans}

    store = MemoryStore()
    lancedb_orphans = get_orphan_lancedb_ids(store, valid_sqlite_ids)

    logger.info("Found %d orphaned expression rows in SQLite.", len(orphans))
    if orphans:
        by_source = Counter((stype, sid) for _, stype, sid in orphans)
        logger.info("Breakdown by deleted source:")
        for (stype, sid), cnt in by_source.most_common(15):
            logger.info("  source_type='%s', source_id=%s: %d expressions", stype, sid, cnt)

    logger.info("Found %d orphaned vector nodes in LanceDB.", len(lancedb_orphans))

    logger.info(
        "\nSUMMARY: This run was a DRY-RUN. To permanently purge these %d SQLite rows and %d LanceDB vectors, run with --execute.",
        len(orphans),
        len(lancedb_orphans),
    )


def run_execute(batch_size: int = 500):
    logger.info("=== STARTING EXECUTE MODE (BATCHED PURGE) ===")
    with get_session() as session:
        orphans = get_orphan_sqlite_ids(session)
        valid_sqlite_ids = {
            r[0]
            for r in session.query(ExpressionRecord.id).all()
        } - {eid for eid, _, _ in orphans}

    orphan_sqlite_ids = [eid for eid, _, _ in orphans]
    store = MemoryStore()
    lancedb_orphans = get_orphan_lancedb_ids(store, valid_sqlite_ids)

    logger.info(
        "Targeting %d SQLite rows and %d LanceDB nodes for deletion (batch size: %d)...",
        len(orphan_sqlite_ids),
        len(lancedb_orphans),
        batch_size,
    )

    # 1. Purge LanceDB in batches
    lancedb_deleted = 0
    for i in range(0, len(lancedb_orphans), batch_size):
        batch = lancedb_orphans[i : i + batch_size]
        ids_str = ", ".join(str(eid) for eid in batch)
        predicate = f"expression_id IN ({ids_str})"
        try:
            store.table.delete(predicate)
            lancedb_deleted += len(batch)
            logger.info(
                "LanceDB batch progress: %d / %d deleted",
                lancedb_deleted,
                len(lancedb_orphans),
            )
        except Exception as e:
            logger.error("Error deleting LanceDB batch at index %d: %s", i, e)
            break

    # 2. Purge SQLite in batches within separate transactions
    sqlite_deleted = 0
    for i in range(0, len(orphan_sqlite_ids), batch_size):
        batch = orphan_sqlite_ids[i : i + batch_size]
        with get_session() as session:
            try:
                # Also clean any dangling relations
                session.query(ExpressionRelation).filter(
                    (ExpressionRelation.from_expression_id.in_(batch))
                    | (ExpressionRelation.to_expression_id.in_(batch))
                ).delete(synchronize_session=False)

                session.query(ExpressionRecord).filter(
                    ExpressionRecord.id.in_(batch)
                ).delete(synchronize_session=False)
                session.commit()
                sqlite_deleted += len(batch)
                logger.info(
                    "SQLite batch progress: %d / %d deleted",
                    sqlite_deleted,
                    len(orphan_sqlite_ids),
                )
            except Exception as e:
                session.rollback()
                logger.error("Error deleting SQLite batch at index %d: %s", i, e)
                break

    logger.info(
        "=== PURGE COMPLETE ===\nDeleted %d LanceDB vectors and %d SQLite expression rows.",
        lancedb_deleted,
        sqlite_deleted,
    )


def main():
    parser = argparse.ArgumentParser(description="Purge orphaned expressions and vectors.")
    parser.add_argument(
        "--execute",
        "--confirm",
        dest="execute",
        action="store_true",
        help="Perform actual deletion. Without this flag, runs in dry-run review mode.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Number of records to delete per transaction/batch (default: 500).",
    )
    args = parser.parse_args()

    if args.execute:
        run_execute(batch_size=args.batch_size)
    else:
        run_review()


if __name__ == "__main__":
    main()
