from unittest.mock import MagicMock, patch

import pytest

from audiobench.daemon.intelligence.blind_spot_detector import BlindSpotDetector


@pytest.mark.asyncio
@patch("audiobench.daemon.intelligence.blind_spot_detector._get_store")
@patch("audiobench.daemon.intelligence.blind_spot_detector.get_calibration_tracker")
async def test_region_classified_as_blind_spot(mock_get_tracker, mock_get_store):
    store = MagicMock()
    mock_get_store.return_value = store

    tracker = MagicMock()
    # 5 inferences, 1 confirm, 4 rejects -> confirm_rate = (1+1)/(5+2) = 2/7 = 0.285 < 0.30
    tracker.stats = {"region1": MagicMock(total_inferences=5, confirm_rate=0.285)}
    mock_get_tracker.return_value = tracker

    detector = BlindSpotDetector()
    await detector.run()

    store.add_expression.assert_called_once()
    args, kwargs = store.add_expression.call_args
    assert "Blind spot detected in region region1" in kwargs["content"]

@pytest.mark.asyncio
@patch("audiobench.daemon.intelligence.blind_spot_detector._get_store")
@patch("audiobench.daemon.intelligence.blind_spot_detector.get_calibration_tracker")
async def test_insufficient_sample_not_classified(mock_get_tracker, mock_get_store):
    store = MagicMock()
    mock_get_store.return_value = store

    tracker = MagicMock()
    # 3 inferences, rate=0.2 < 0.3 but < 5 samples
    tracker.stats = {"region1": MagicMock(total_inferences=3, confirm_rate=0.2)}
    mock_get_tracker.return_value = tracker

    detector = BlindSpotDetector()
    await detector.run()

    store.add_expression.assert_not_called()

@pytest.mark.asyncio
@patch("audiobench.daemon.intelligence.blind_spot_detector._get_store")
@patch("audiobench.daemon.intelligence.blind_spot_detector.get_calibration_tracker")
async def test_blind_spot_writes_calibration_expression(mock_get_tracker, mock_get_store):
    store = MagicMock()
    mock_get_store.return_value = store

    tracker = MagicMock()
    tracker.stats = {"region2": MagicMock(total_inferences=10, confirm_rate=0.1)}
    mock_get_tracker.return_value = tracker

    detector = BlindSpotDetector()
    await detector.run()

    store.add_expression.assert_called_once()
    args, kwargs = store.add_expression.call_args
    assert kwargs["source_type"] == "daemon_calibration"
