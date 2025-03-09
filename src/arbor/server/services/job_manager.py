import uuid
from enum import Enum
from arbor.server.api.models.schemas import FineTuneRequest

class JobStatus(Enum):
  PENDING = "pending"
  QUEUED = "queued"
  RUNNING = "running"
  COMPLETED = "completed"
  FAILED = "failed"

class Job:
  def __init__(self, id: str, status: JobStatus):
    self.id = id
    self.status = status

class JobManager:
  def __init__(self):
    self.jobs = {}

  def get_job_status(self, job_id: str):
    return self.jobs[job_id].status

  def create_job(self):
    job = Job(id=str(uuid.uuid4()), status=JobStatus.PENDING)
    self.jobs[job.id] = job
    return job
