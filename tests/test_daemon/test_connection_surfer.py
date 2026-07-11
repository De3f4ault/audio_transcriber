import pytest
import numpy as np
import json
from unittest.mock import MagicMock, patch
from audiobench.daemon.intelligence.connection_surfer import ConnectionSurfer
from audiobench.storage.models import PendingRelation

@pytest.fixture
def mock_store():
    store = MagicMock()
    store.written = []
    def mock_add(source_type, content, inference_status=None, **kwargs):
        expr = MagicMock()
        expr.id = 200 + len(store.written)
        store.written.append({
            "source_type": source_type,
            "content": content,
            "inference_status": inference_status
        })
        return expr
    store.add_expression.side_effect = mock_add
    return store

@pytest.fixture
def mock_session():
    session = MagicMock()
    session.pending_relations = []
    def mock_add(obj):
        if isinstance(obj, PendingRelation):
            session.pending_relations.append(obj)
    session.add.side_effect = mock_add
    
    def create_mock_execute(calib_content=None, pairs_data=None):
        def mock_execute(*args, **kwargs):
            query = str(args[0])
            res = MagicMock()
            if "daemon_calibration" in query:
                res.fetchone.return_value = (calib_content,) if calib_content else None
            elif "SELECT e.id" in query or "JOIN expression_segment_map" in query or "JOIN segments" in query:
                # Mock the 200 items fetch
                map_res = MagicMock()
                map_res.all.return_value = pairs_data or []
                res.mappings.return_value = map_res
            return res
        return mock_execute
        
    session.create_mock_execute = create_mock_execute
    
    ctx = MagicMock()
    ctx.__enter__.return_value = session
    return ctx

@pytest.mark.asyncio
@patch("audiobench.daemon.intelligence.connection_surfer.get_session")
@patch("audiobench.daemon.intelligence.connection_surfer._get_store")
async def test_cross_file_pair_above_threshold_creates_pending_relation(mock_get_store, mock_get_session, mock_store, mock_session):
    # Two expressions from different files (audio_file_id 1 and 2), highly similar
    data = [
        {"id": 1, "audio_file_id": 1, "content": "A test"},
        {"id": 2, "audio_file_id": 2, "content": "B test"}
    ]
    
    session = mock_session.__enter__()
    session.execute.side_effect = session.create_mock_execute(None, data)
    
    # 0.9 similarity
    vecs = {1: [1.0, 0.0], 2: [0.9, 0.435889]} # cos_sim = 0.9
    mock_store.get_vectors.return_value = vecs
    
    mock_get_store.return_value = mock_store
    mock_get_session.return_value = mock_session
    
    surfer = ConnectionSurfer()
    await surfer.run()
    
    assert len(session.pending_relations) == 1
    assert session.pending_relations[0].from_expression_id == 1
    assert session.pending_relations[0].to_expression_id_hint == 2
    
    exprs = [x for x in mock_store.written if x["source_type"] == "potential_relation"]
    assert len(exprs) == 1
    assert "Potential connection detected" in exprs[0]["content"]

@pytest.mark.asyncio
@patch("audiobench.daemon.intelligence.connection_surfer.get_session")
@patch("audiobench.daemon.intelligence.connection_surfer._get_store")
async def test_no_inference_for_same_file_pairs(mock_get_store, mock_get_session, mock_store, mock_session):
    # Same file
    data = [
        {"id": 1, "audio_file_id": 1, "content": "A test"},
        {"id": 2, "audio_file_id": 1, "content": "B test"}
    ]
    
    session = mock_session.__enter__()
    session.execute.side_effect = session.create_mock_execute(None, data)
    
    vecs = {1: [1.0, 0.0], 2: [1.0, 0.0]} # sim = 1.0 > 0.85
    mock_store.get_vectors.return_value = vecs
    
    mock_get_store.return_value = mock_store
    mock_get_session.return_value = mock_session
    
    surfer = ConnectionSurfer()
    await surfer.run()
    
    assert len(session.pending_relations) == 0
    assert len(mock_store.written) == 0

@pytest.mark.asyncio
@patch("audiobench.daemon.intelligence.connection_surfer.get_session")
@patch("audiobench.daemon.intelligence.connection_surfer._get_store")
async def test_pair_normalisation_prevents_duplicate(mock_get_store, mock_get_session, mock_store, mock_session):
    data = [
        {"id": 1, "audio_file_id": 1, "content": "A test"},
        {"id": 2, "audio_file_id": 2, "content": "B test"}
    ]
    
    session = mock_session.__enter__()
    calib = json.dumps({"surfer_seen_pairs": [[1, 2]]})
    session.execute.side_effect = session.create_mock_execute(calib, data)
    
    vecs = {1: [1.0, 0.0], 2: [1.0, 0.0]} # sim = 1.0 > 0.85
    mock_store.get_vectors.return_value = vecs
    
    mock_get_store.return_value = mock_store
    mock_get_session.return_value = mock_session
    
    surfer = ConnectionSurfer()
    await surfer.run()
    
    assert len(session.pending_relations) == 0
    assert len([x for x in mock_store.written if x["source_type"] == "potential_relation"]) == 0

@pytest.mark.asyncio
@patch("audiobench.daemon.intelligence.connection_surfer.get_session")
@patch("audiobench.daemon.intelligence.connection_surfer._get_store")
async def test_max_per_session_respected(mock_get_store, mock_get_session, mock_store, mock_session):
    # create 15 expressions, each from a different file, all identical vectors
    data = [{"id": i, "audio_file_id": i, "content": "test"} for i in range(15)]
    vecs = {i: [1.0, 0.0] for i in range(15)}
    
    session = mock_session.__enter__()
    session.execute.side_effect = session.create_mock_execute(None, data)
    mock_store.get_vectors.return_value = vecs
    
    mock_get_store.return_value = mock_store
    mock_get_session.return_value = mock_session
    
    surfer = ConnectionSurfer()
    await surfer.run()
    
    assert len(session.pending_relations) == 10
    assert len([x for x in mock_store.written if x["source_type"] == "potential_relation"]) == 10
