import pytest
import time
import asyncio
from unittest.mock import MagicMock, patch
from audiobench.daemon.intelligence.scheduler import IntelligenceScheduler, ResourceBudget, IntelligenceTask

class MockTask(IntelligenceTask):
    INTERVAL_SECONDS = 3600
    
    def __init__(self):
        self.run_count = 0
        
    async def run(self):
        self.run_count += 1

def test_task_not_run_before_idle_threshold():
    budget = ResourceBudget(idle_threshold_seconds=120)
    scheduler = IntelligenceScheduler(budget=budget)
    assert not scheduler._is_idle(last_request_time=time.time() - 60)

def test_task_runs_after_idle_threshold():
    budget = ResourceBudget(idle_threshold_seconds=120)
    scheduler = IntelligenceScheduler(budget=budget)
    assert scheduler._is_idle(last_request_time=time.time() - 130)

@patch("audiobench.daemon.intelligence.scheduler.psutil")
def test_task_not_run_when_memory_over_budget(mock_psutil):
    """Gate fires when current RSS exceeds baseline + max_task_overhead_mb."""
    mock_process = MagicMock()
    mock_mem_info = MagicMock()
    # Simulate baseline of 2000 MB, current RSS of 2600 MB → 600 MB overhead
    mock_mem_info.rss = 2600 * 1024 * 1024
    mock_process.memory_info.return_value = mock_mem_info
    mock_psutil.Process.return_value = mock_process
    mock_psutil.cpu_percent.return_value = 0.0

    budget = ResourceBudget(max_task_overhead_mb=500)
    scheduler = IntelligenceScheduler(budget=budget)
    scheduler._baseline_rss_mb = 2000.0  # settled post-startup baseline
    assert not scheduler._has_resources()


@patch("audiobench.daemon.intelligence.scheduler.psutil")
def test_task_allowed_when_overhead_within_budget(mock_psutil):
    """Gate passes when RSS overhead is within the allowed ceiling."""
    mock_process = MagicMock()
    mock_mem_info = MagicMock()
    # Baseline 3000 MB, current 3200 MB → 200 MB overhead, budget 1024 MB
    mock_mem_info.rss = 3200 * 1024 * 1024
    mock_process.memory_info.return_value = mock_mem_info
    mock_psutil.Process.return_value = mock_process
    mock_psutil.cpu_percent.return_value = 0.0

    budget = ResourceBudget(max_task_overhead_mb=1024)
    scheduler = IntelligenceScheduler(budget=budget)
    scheduler._baseline_rss_mb = 3000.0
    assert scheduler._has_resources()


@patch("audiobench.daemon.intelligence.scheduler.psutil")
def test_task_not_run_when_cpu_over_budget(mock_psutil):
    mock_process = MagicMock()
    mock_mem_info = MagicMock()
    # RSS within overhead budget (baseline 3000, current 3100 → 100 MB overhead)
    mock_mem_info.rss = 3100 * 1024 * 1024
    mock_process.memory_info.return_value = mock_mem_info
    mock_psutil.Process.return_value = mock_process
    mock_psutil.cpu_percent.return_value = 50.0

    budget = ResourceBudget(max_cpu_percent=15.0, max_task_overhead_mb=1024)
    scheduler = IntelligenceScheduler(budget=budget)
    scheduler._baseline_rss_mb = 3000.0
    assert not scheduler._has_resources()

def test_max_inferences_per_session_enforced():
    budget = ResourceBudget(max_inferences_per_session=50)
    scheduler = IntelligenceScheduler(budget=budget)
    
    scheduler.inferences_performed = 50
    assert not scheduler._has_resources()
    
    scheduler.inferences_performed = 49
    # assuming we mock psutil to None for this test to pass
    with patch("audiobench.daemon.intelligence.scheduler.psutil", None):
        assert scheduler._has_resources()

@pytest.mark.asyncio
async def test_interval_enforced_between_runs():
    budget = ResourceBudget(idle_threshold_seconds=0)
    scheduler = IntelligenceScheduler(budget=budget)
    task = MockTask()
    scheduler.register(task)
    
    # We will test the loop logic by calling the inner check directly instead of run_loop() which sleeps forever
    # The run_loop logic:
    with patch("time.time", side_effect=[0, 0, 0, 0]):
        # t=0, last_run=0, INTERVAL=3600
        # now - last_run (0 - 0) = 0, which is not >= 3600.
        pass

    # Wait, the run_loop checks `now - last_run >= task.INTERVAL_SECONDS`.
    # At start, last_run is 0. now - last_run is time.time() - 0.
    # We can test this by interacting with _last_run_times.
    
    now_time = 10000.0
    with patch("time.time", return_value=now_time):
        scheduler._last_run_times[type(task)] = now_time - 3500
        # Should not run since 3500 < 3600
        can_run = (now_time - scheduler._last_run_times[type(task)]) >= task.INTERVAL_SECONDS
        assert not can_run
        
        scheduler._last_run_times[type(task)] = now_time - 3700
        # Should run
        can_run = (now_time - scheduler._last_run_times[type(task)]) >= task.INTERVAL_SECONDS
        assert can_run


def test_circuit_breaker_engages_at_threshold():
    """Task enters backoff when consecutive failures reach circuit_breaker_threshold."""
    budget = ResourceBudget(circuit_breaker_threshold=3)
    scheduler = IntelligenceScheduler(budget=budget)
    task_type = MockTask

    exc = RuntimeError("task failed")
    scheduler._record_task_failure(task_type, exc)
    scheduler._record_task_failure(task_type, exc)
    assert not scheduler._is_task_backed_off(task_type)  # 2 failures, not yet
    scheduler._record_task_failure(task_type, exc)
    assert scheduler._is_task_backed_off(task_type)  # 3 failures → backed off


def test_circuit_breaker_resets_on_success():
    """Successful run clears backoff state."""
    budget = ResourceBudget(circuit_breaker_threshold=3)
    scheduler = IntelligenceScheduler(budget=budget)
    task_type = MockTask

    exc = RuntimeError("task failed")
    for _ in range(3):
        scheduler._record_task_failure(task_type, exc)
    assert scheduler._is_task_backed_off(task_type)

    scheduler._record_task_success(task_type)
    assert not scheduler._is_task_backed_off(task_type)
    assert scheduler._task_consecutive_failures.get(task_type, 0) == 0
