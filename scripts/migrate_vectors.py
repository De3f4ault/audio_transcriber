"""
migrate_vectors.py
==================
One-time migration: delete all Tier 1 and Tier 2 vectors from LanceDB.

Background
----------
The original ingestion pipeline wrote vectors for all three chunk tiers:
  - Tier 1: full cleaned transcript    (source_type = audio_transcript, no inbound SOURCE rels)
  - Tier 2: parent paragraph groups    (target of Tier 3's SOURCE relation)
  - Tier 3: individual sentences       (leaf nodes — these are the ONLY ones we want in LanceDB)

Because Tier 1 and Tier 2 share the same source_type and have similar
(or identical due to 512-token truncation) vectors, they duplicate results
and pollute semantic search rankings.

This script:
  1. Identifies all Tier 1 + Tier 2 expression IDs from SQLite.
  2. Deletes their vectors from LanceDB via the daemon client.
  3. Leaves the SQLite ExpressionRecord rows intact — they are needed for
     graph traversal during search (parent-child expansion).

Run once:
    python scripts/migrate_vectors.py

After the migration, re-run any search to confirm duplicates are gone.
"""

import sys
from pathlib import Path

# Make sure the package is importable when run from the project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from audiobench.core.db_session import get_session
from audiobench.core.logger_factory import get_logger
from audiobench.daemon.factory import get_daemon_client
from audiobench.storage.models import ExpressionRecord, ExpressionRelation

logger = get_logger("migrate_vectors")


def get_tier1_and_tier2_ids(session) -> set[int]:
    """
    Identify all Tier 1 and Tier 2 expression IDs.

    Tier 3 sentence chunks are leaf nodes — they have an outbound SOURCE
    relation (pointing up to their Tier 2 parent) but NO inbound SOURCE
    relations (nothing points to them as a parent).

    Tier 1 and Tier 2 are the non-leaf nodes — they appear as targets
    (to_expression_id) of at least one SOURCE relation.

    We collect all expression IDs that appear as a target of any SOURCE
    relation. These are the Tier 1 and Tier 2 nodes we want out of LanceDB.
    """
    target_ids = set(
        row[0]
        for row in session.query(ExpressionRelation.to_expression_id)
        .filter_by(relation_type="source")
        .all()
    )
    return target_ids


def main() -> None:
    print("=" * 60)
    print("AudioBench Vector Migration: Remove Tier 1 & Tier 2 vectors")
    print("=" * 60)

    try:
        daemon = get_daemon_client()
    except Exception as e:
        print(f"\n[ERROR] Could not connect to daemon: {e}")
        print("Make sure the AudioBench daemon is running before migrating.")
        sys.exit(1)

    with get_session() as session:
        ids_to_delete = get_tier1_and_tier2_ids(session)

    total = len(ids_to_delete)
    print(f"\nFound {total} Tier 1 / Tier 2 expression IDs to remove from LanceDB.")

    if total == 0:
        print("Nothing to do. LanceDB is already clean.")
        return

    print("Starting deletion...\n")
    deleted = 0
    failed = 0

    for i, expr_id in enumerate(sorted(ids_to_delete), 1):
        try:
            daemon.delete(expr_id)
            deleted += 1
            if i % 50 == 0 or i == total:
                print(f"  [{i}/{total}] Deleted expression ID {expr_id}")
        except Exception as e:
            failed += 1
            logger.warning("Failed to delete expression %d from LanceDB: %s", expr_id, e)

    print("\n" + "=" * 60)
    print(f"Migration complete.")
    print(f"  Deleted : {deleted}")
    print(f"  Failed  : {failed}")
    print(f"  SQLite records are untouched (graph intact for parent-walk).")
    print("=" * 60)

    if failed > 0:
        print(
            "\n[WARNING] Some deletions failed. These IDs may not exist in LanceDB\n"
            "(e.g. they were never vectorised, or a previous run already removed them).\n"
            "This is safe to ignore if the count is small."
        )


if __name__ == "__main__":
    main()
