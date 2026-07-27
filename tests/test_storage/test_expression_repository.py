"""Tests for the ExpressionRepository."""

from audiobench.memory.enums import InferenceStatus, RelationType, SourceType
from audiobench.storage.expression_repository import ExpressionRepository


def test_register_expression(test_db):
    repo = ExpressionRepository()

    expr = repo.register(
        content="Test content",
        source_type=SourceType.SYSTEM_INFERENCE.value,
        session_type="chat",
        speaker="assistant",
    )

    assert expr.id is not None
    assert expr.content == "Test content"
    assert expr.source_type == "system_inference"
    assert expr.speaker == "assistant"

    # Retrieve and verify
    retrieved = repo.get_by_id(expr.id)
    assert retrieved is not None
    assert retrieved.content == "Test content"


def test_link_expressions(test_db):
    repo = ExpressionRepository()

    expr1 = repo.register(content="Source node", source_type=SourceType.ASK_QUERY.value)
    expr2 = repo.register(content="Derived node", source_type=SourceType.ASK_ANSWER.value)

    relation = repo.link(
        from_id=expr2.id,
        to_id=expr1.id,
        relation_type=RelationType.SOURCE.value,
    )

    assert relation.id is not None
    assert relation.from_expression_id == expr2.id
    assert relation.to_expression_id == expr1.id
    assert relation.relation_type == "source"


def test_walk_to_parent(test_db):
    repo = ExpressionRepository()

    # Hierarchy: parent <- child <- grandchild
    parent = repo.register(content="Parent", source_type=SourceType.AUDIO_TRANSCRIPT.value)
    child = repo.register(content="Child", source_type=SourceType.TRANSCRIPT_SEGMENT.value)
    grandchild = repo.register(content="Grandchild", source_type=SourceType.SYSTEM_INFERENCE.value)

    # Link child to parent (child -> parent)
    repo.link(from_id=child.id, to_id=parent.id, relation_type=RelationType.SOURCE.value)

    # Link grandchild to child (grandchild -> child)
    repo.link(from_id=grandchild.id, to_id=child.id, relation_type=RelationType.SOURCE.value)

    # Walk up from grandchild gives child
    found_child = repo.walk_to_parent(grandchild.id)
    assert found_child is not None
    assert found_child.id == child.id

    # Walk up from child gives parent
    found_parent = repo.walk_to_parent(child.id)
    assert found_parent is not None
    assert found_parent.id == parent.id

    # Walk up from parent gives None
    found_none = repo.walk_to_parent(parent.id)
    assert found_none is None


def test_update_inference_status(test_db):
    repo = ExpressionRepository()

    expr = repo.register(
        content="An inference",
        source_type=SourceType.SYSTEM_INFERENCE.value,
    )

    assert expr.inference_status is None

    # Update status
    success = repo.update_inference_status(expr.id, InferenceStatus.ACTIVE.value)
    assert success is True

    # Verify
    updated = repo.get_by_id(expr.id)
    assert updated is not None
    assert updated.inference_status == "active"

    # Test update non-existent
    success = repo.update_inference_status(999999, InferenceStatus.DEPRECATED.value)
    assert success is False


def test_register_accepts_work_id_kwarg(test_db):
    """Regression: sweep loop passes work_id= to register(); a missing param
    causes a TypeError that silently kills the entire sweep batch.

    If this test fails, the fix is to add `work_id: int | None = None` to the
    `register()` signature and pass it through to ExpressionRecord.
    """
    repo = ExpressionRepository()

    # Must not raise TypeError even when work_id is supplied
    expr = repo.register(
        content="Test utterance for sweep",
        source_type=SourceType.AUDIO_TRANSCRIPT.value,
        source_id=42,
        work_id=7,
    )

    assert expr is not None
    assert expr.id is not None
    assert expr.work_id == 7


def test_register_batch_cross_transcript_isolation(test_db):
    """Regression: identical content in two different transcripts must produce
    two independent ExpressionRecords, not one shared record.

    Before Track 2 fix: register_batch() deduped on (content_hash, source_type)
    only, so tx_B would silently reuse tx_A's expression (wrong source_id).
    LanceDB's merge_insert would then overwrite tx_A's metadata with tx_B's,
    actively corrupting the older record's search attribution.
    """
    repo = ExpressionRepository()

    items_a = [
        {
            "content": "Okay.",
            "source_type": SourceType.AUDIO_TRANSCRIPT.value,
            "source_id": 100,
        }
    ]
    items_b = [
        {
            "content": "Okay.",
            "source_type": SourceType.AUDIO_TRANSCRIPT.value,
            "source_id": 200,
        }
    ]

    results_a = repo.register_batch(items_a)
    results_b = repo.register_batch(items_b)

    assert len(results_a) == 1
    assert len(results_b) == 1

    expr_a = results_a[0]
    expr_b = results_b[0]

    # Must be distinct rows in SQLite — not the same ExpressionRecord
    assert expr_a.id != expr_b.id, (
        "Identical content across two different source_ids must NOT be deduped "
        "into a single ExpressionRecord. Found same id=%d for both." % expr_a.id
    )
    assert expr_a.source_id == 100
    assert expr_b.source_id == 200


def test_register_batch_within_transcript_idempotency(test_db):
    """Within the same transcript, identical content IS correctly deduped.

    This covers the sweep-retry path: if the daemon crashes mid-sweep and
    re-processes the same transcript, it must not insert duplicate rows.

    Implementation note: register_batch() deduplicates *within* the batch
    itself at Step 1 using content_hash alone (before source_id is considered),
    so two identical items in a single call always collapse to one result.
    Calling register_batch() a second time with the same items also returns
    that same ExpressionRecord via the DB lookup path.

    Accepted tradeoff: two genuinely distinct moments in one transcript with
    the exact same text collapse to one ExpressionRecord. This is a known,
    logged edge case — short affirmations losing one search-hit destination
    is acceptable; creating phantom duplicates on retry is not.
    """
    repo = ExpressionRepository()

    items = [
        {
            "content": "Right.",
            "source_type": SourceType.AUDIO_TRANSCRIPT.value,
            "source_id": 100,
        },
        {
            "content": "Right.",
            "source_type": SourceType.AUDIO_TRANSCRIPT.value,
            "source_id": 100,
        },
    ]

    # First call: within-batch dedup collapses two items → one row created
    results_first = repo.register_batch(items)
    assert len(results_first) == 1, (
        "register_batch dedupes within the batch — two identical items must yield one result"
    )
    expr_id = results_first[0].id

    # Second call (retry simulation): same items again → same row returned, no duplicate created
    results_second = repo.register_batch(items)
    assert len(results_second) == 1
    assert results_second[0].id == expr_id, (
        "Retry of the same batch must return the existing ExpressionRecord, not create a duplicate"
    )


