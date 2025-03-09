import uuid
from enum import Enum
import logging
from datetime import datetime

class JobStatus(Enum):
  PENDING = "pending"
  QUEUED = "queued"
  RUNNING = "running"
  COMPLETED = "completed"
  FAILED = "failed"

class JobLogHandler(logging.Handler):
  def __init__(self, job):
    super().__init__()
    self.job = job

  def emit(self, record):
    log_entry = {
      'timestamp': datetime.fromtimestamp(record.created).isoformat(),
      'level': record.levelname,
      'message': record.getMessage()
    }
    self.job.logs.append(log_entry)

class Job:
  def __init__(self, id: str, status: JobStatus):
    self.id = id
    self.status = status
    self.logs = []
    self.logger = None
    self.log_handler = None

  def setup_logger(self, logger_name: str = None) -> logging.Logger:
    """Sets up logging for the job with a dedicated handler."""
    if logger_name is None:
      logger_name = f"job_{self.id}"

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Create and setup handler if not already exists
    if self.log_handler is None:
      handler = JobLogHandler(self)
      formatter = logging.Formatter('%(message)s')
      handler.setFormatter(formatter)
      logger.addHandler(handler)
      self.log_handler = handler

    self.logger = logger
    return logger

  def cleanup_logger(self):
    """Removes the job's logging handler."""
    if self.logger and self.log_handler:
      self.logger.removeHandler(self.log_handler)
      self.log_handler = None
      self.logger = None

class JobManager:
  def __init__(self):
    self.jobs = {}

  def get_job_status(self, job_id: str):
    if job_id not in self.jobs:
      raise ValueError(f"Job {job_id} not found")
    return self.jobs[job_id].status

  def create_job(self):
    job = Job(id=str(uuid.uuid4()), status=JobStatus.PENDING)
    self.jobs[job.id] = job
    return job
