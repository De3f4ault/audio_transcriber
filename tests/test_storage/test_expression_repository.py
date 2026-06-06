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
