from functools import lru_cache
from arbor.server.services.file_manager import FileManager
from arbor.server.services.job_manager import JobManager
from arbor.server.services.training_manager import TrainingManager

@lru_cache()
def get_file_manager() -> FileManager:
    return FileManager()

@lru_cache()
def get_job_manager() -> JobManager:
    return JobManager()

@lru_cache()
def get_training_manager() -> TrainingManager:
    return TrainingManager()
