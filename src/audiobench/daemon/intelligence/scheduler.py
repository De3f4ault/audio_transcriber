import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger("audiobench.daemon.intelligence")

@dataclass
class ResourceBudget:
    max_memory_mb: int = 4096        # Accommodates 4 locally loaded PyTorch models + LanceDB
    max_cpu_percent: float = 80.0    # Run if overall system CPU is under 80%
    idle_threshold_seconds: int = 120
    max_inferences_per_session: int = 50

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

    def register(self, task: IntelligenceTask) -> None:
        self.tasks.append(task)
        logger.info("Registered intelligence task: %s", type(task).__name__)

    def _is_idle(self, last_request_time: float) -> bool:
        if time.time() - last_request_time < self.budget.idle_threshold_seconds:
            return False
        return True

    def _has_resources(self) -> bool:
        if self.inferences_performed >= self.budget.max_inferences_per_session:
            return False
            
        if psutil is not None:
            # Check RSS memory of current process
            if not hasattr(self, "_process"):
                self._process = psutil.Process()
            mem_info = self._process.memory_info()
            rss_mb = mem_info.rss / (1024 * 1024)
            if rss_mb > self.budget.max_memory_mb:
                logger.warning("Intelligence task aborted: RSS %.1fMB > %.1fMB", rss_mb, self.budget.max_memory_mb)
                return False
                
            # Check overall CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent > self.budget.max_cpu_percent:
                logger.debug("Intelligence task delayed: CPU usage %.1f%% > %.1f%%", cpu_percent, self.budget.max_cpu_percent)
                return False
                
        return True

    async def run_loop(self) -> None:
        """The main loop for the intelligence scheduler."""
        from audiobench.daemon import server
        
        # Wait a bit after startup
        await asyncio.sleep(60)
        
        while True:
            try:
                # Poll server._last_request_time
                last_req = getattr(server, "_last_request_time", time.time())
                
                if self._is_idle(last_req) and self._has_resources():
                    now = time.time()
                    for task in self.tasks:
                        task_type = type(task)
                        last_run = self._last_run_times.get(task_type, 0)
                        if now - last_run >= task.INTERVAL_SECONDS:
                            logger.info("Executing intelligence task: %s", task_type.__name__)
                            await task.run()
                            self._last_run_times[task_type] = time.time()
                            self.inferences_performed += 1
                            break # Only run one task per check
            except Exception as e:
                logger.error("Error in intelligence scheduler loop: %s", e)
                
            await asyncio.sleep(30)
