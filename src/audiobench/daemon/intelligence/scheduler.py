import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger("audiobench.daemon.intelligence")

@dataclass
class ResourceBudget:
    max_task_overhead_mb: int = 1024  # Max RSS *above baseline* allowed for task work.
    # The absolute ceiling (max_memory_mb) is gone — replaced by an overhead-relative
    # gate measured against the post-startup baseline.  Rationale (2026-07-20):
    # The daemon loads 4 PyTorch models that consume ~3 GB RSS at rest.  An
    # absolute 4096 MB ceiling leaves only 1 GB of headroom, which is fine for
    # task work but caused the gate to fire during sweep batches that legitimately
    # push RSS up temporarily.  The overhead-relative gate ignores the permanent
    # model footprint and only reacts to transient work above the settled baseline.
    max_cpu_percent: float = 80.0    # Run if overall system CPU is under 80%
    idle_threshold_seconds: int = 120
    max_inferences_per_session: int = 50
    # Circuit breaker: how many consecutive task failures before exponential backoff.
    # Named explicitly as a starting guess, not a derived value.  Adjust if observed
    # failure patterns differ from this assumption.
    circuit_breaker_threshold: int = 3

@runtime_checkable
class IntelligenceTask(Protocol):
    INTERVAL_SECONDS: int
    
    async def run(self) -> None:
        """Execute the intelligence task."""
        ...

class IntelligenceScheduler:
    def __init__(self, budget: ResourceBudget | None = None):
        self.budget = budget or ResourceBudget()
        self.tasks: list[IntelligenceTask] = []
        self._last_run_times: dict[type, float] = {}
        self.inferences_performed = 0
        # Baseline RSS measured after the startup warmup sleep, before any task
        # has run.  The gate checks overhead above this value, not absolute RSS.
        self._baseline_rss_mb: float | None = None
        # Per-task consecutive failure counters for the circuit breaker.
        self._task_consecutive_failures: dict[type, int] = {}
        self._task_backoff_until: dict[type, float] = {}

    def register(self, task: IntelligenceTask) -> None:
        self.tasks.append(task)
        logger.info("Registered intelligence task: %s", type(task).__name__)

    def _is_idle(self, last_request_time: float) -> bool:
        if time.time() - last_request_time < self.budget.idle_threshold_seconds:
            return False
        return True

    def _measure_rss_mb(self) -> float | None:
        """Return current process RSS in MB, or None if psutil is unavailable."""
        if psutil is None:
            return None
        if not hasattr(self, "_process"):
            self._process = psutil.Process()
        return self._process.memory_info().rss / (1024 * 1024)

    def _has_resources(self) -> bool:
        if self.inferences_performed >= self.budget.max_inferences_per_session:
            return False

        if psutil is not None:
            rss_mb = self._measure_rss_mb()
            if rss_mb is not None:
                # Gate on overhead above the settled post-startup baseline.
                # If baseline hasn't been measured yet (shouldn't happen — it's
                # set in run_loop before the first gate check), fall back to a
                # generous absolute ceiling so the task isn't blocked forever.
                if self._baseline_rss_mb is not None:
                    overhead_mb = rss_mb - self._baseline_rss_mb
                    if overhead_mb > self.budget.max_task_overhead_mb:
                        logger.warning(
                            "Intelligence task aborted: RSS overhead %.1fMB > %.1fMB "
                            "(current=%.1fMB, baseline=%.1fMB)",
                            overhead_mb, self.budget.max_task_overhead_mb,
                            rss_mb, self._baseline_rss_mb,
                        )
                        return False
                else:
                    # Fallback: absolute ceiling while baseline is unavailable.
                    if rss_mb > 6144:
                        logger.warning(
                            "Intelligence task aborted: RSS %.1fMB > 6144MB (no baseline yet)",
                            rss_mb,
                        )
                        return False

            # Check overall CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent > self.budget.max_cpu_percent:
                logger.debug(
                    "Intelligence task delayed: CPU usage %.1f%% > %.1f%%",
                    cpu_percent, self.budget.max_cpu_percent,
                )
                return False

        return True

    def _is_task_backed_off(self, task_type: type) -> bool:
        """Return True if this task type is in exponential backoff."""
        backoff_until = self._task_backoff_until.get(task_type, 0)
        return time.time() < backoff_until

    def _record_task_success(self, task_type: type) -> None:
        """Reset the circuit breaker for a task after a successful run."""
        self._task_consecutive_failures[task_type] = 0
        self._task_backoff_until.pop(task_type, None)

    def _record_task_failure(self, task_type: type, exc: Exception) -> None:
        """Increment the failure counter and engage exponential backoff if threshold hit."""
        n = self._task_consecutive_failures.get(task_type, 0) + 1
        self._task_consecutive_failures[task_type] = n
        logger.error(
            "Intelligence task %s failed (consecutive failure %d): %s",
            task_type.__name__, n, exc,
        )
        if n >= self.budget.circuit_breaker_threshold:
            # Exponential backoff: 2^(n - threshold) * 30 minutes, capped at 6 hours.
            exponent = n - self.budget.circuit_breaker_threshold
            backoff_seconds = min(2 ** exponent * 1800, 21600)
            self._task_backoff_until[task_type] = time.time() + backoff_seconds
            logger.warning(
                "Intelligence task %s circuit breaker engaged after %d failures — "
                "backing off for %.0f minutes.",
                task_type.__name__, n, backoff_seconds / 60,
            )

    async def run_loop(self) -> None:
        """The main loop for the intelligence scheduler."""
        from audiobench.daemon import server

        # Wait for startup to settle before measuring baseline or running tasks.
        await asyncio.sleep(60)

        # Measure the settled post-startup RSS baseline.  This is the model-loaded
        # resident footprint before any task work has been done.
        rss_mb = self._measure_rss_mb()
        if rss_mb is not None:
            self._baseline_rss_mb = rss_mb
            logger.info(
                "Intelligence scheduler baseline RSS: %.1f MB "
                "(gate will fire at baseline + %d MB = %.1f MB)",
                rss_mb, self.budget.max_task_overhead_mb,
                rss_mb + self.budget.max_task_overhead_mb,
            )
        else:
            logger.warning("Intelligence scheduler: psutil unavailable, RSS gate disabled.")

        while True:
            try:
                # Poll server._last_request_time
                last_req = getattr(server, "_last_request_time", time.time())

                if self._is_idle(last_req) and self._has_resources():
                    now = time.time()
                    for task in self.tasks:
                        task_type = type(task)
                        last_run = self._last_run_times.get(task_type, 0)
                        if now - last_run < task.INTERVAL_SECONDS:
                            continue
                        if self._is_task_backed_off(task_type):
                            backoff_remaining = self._task_backoff_until[task_type] - now
                            logger.debug(
                                "Intelligence task %s skipped (circuit breaker: "
                                "%.0f min remaining).",
                                task_type.__name__, backoff_remaining / 60,
                            )
                            continue
                        logger.info("Executing intelligence task: %s", task_type.__name__)
                        try:
                            await task.run()
                            self._last_run_times[task_type] = time.time()
                            self.inferences_performed += 1
                            self._record_task_success(task_type)
                        except Exception as task_exc:
                            self._record_task_failure(task_type, task_exc)
                        break  # Only run one task per check
            except Exception as e:
                logger.error("Error in intelligence scheduler loop: %s", e)

            await asyncio.sleep(30)
