import pytest
from arbor.server.services.job_manager import JobManager, Job, JobStatus
from arbor.server.core.config import Settings

@pytest.fixture
def test_settings(tmp_path):
    # tmp_path is a Path object that points to a temporary directory
    # It will be automatically cleaned up after tests
    return Settings(
        STORAGE_PATH=str(tmp_path / "test_storage")
    )

@pytest.fixture
def job_manager(test_settings):
    return JobManager(settings=test_settings)

def test_create_job(job_manager):
    job = job_manager.create_job()

    assert isinstance(job, Job)
    assert job.id in job_manager.jobs
    assert job.status == JobStatus.PENDING
    assert job.logs == []
    assert job.logger is None
    assert job.log_handler is None
    assert job.fine_tuned_model is None

def test_job_logging(job_manager):
    job = job_manager.create_job()

    # Test logger setup
    logger = job.setup_logger("test_logger")
    assert job.logger is not None
    assert job.log_handler is not None

    # Test logging functionality
    logger.info("Test message")
    assert len(job.logs) == 1
    log_entry = job.logs[0]
    assert log_entry['level'] == 'INFO'
    assert log_entry['message'] == 'Test message'

    # Test cleanup
    job.cleanup_logger()
    assert job.logger is None
    assert job.log_handler is None

def test_get_job(job_manager):
    job = job_manager.create_job()

    assert job_manager.get_job(job.id).status == JobStatus.PENDING

    with pytest.raises(ValueError):
        job_manager.get_job("nonexistent-id")