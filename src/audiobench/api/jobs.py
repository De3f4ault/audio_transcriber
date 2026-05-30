import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Any

@dataclass
class Job:
    id: str
    status: str = "pending"  # pending, processing, completed, error
    progress: float = 0.0
    segments: List[dict] = field(default_factory=list)
    result: dict | None = None
    error: str | None = None
    is_cancelled: bool = False

class JobManager:
    """In-memory tracking of transcription jobs for the web UI."""
    def __init__(self) -> None:
        self.jobs: Dict[str, Job] = {}

    def create_job(self) -> Job:
        job_id = str(uuid.uuid4())
        job = Job(id=job_id)
        self.jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

job_manager = JobManager()
