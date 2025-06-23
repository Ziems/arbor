import pytest

from arbor.server.core.config import Config
from arbor.server.services.job_manager import (
    Job,
    JobCheckpoint,
    JobEvent,
    JobManager,
    JobStatus,
)


@pytest.fixture
def test_settings(tmp_path):
    # tmp_path is a Path object that points to a temporary directory
    # It will be automatically cleaned up after tests
    return Config(STORAGE_PATH=str(tmp_path / "test_storage"))


@pytest.fixture
def job_manager(test_settings):
    return JobManager(settings=test_settings)


def test_create_job(job_manager):
    job = job_manager.create_job()

    assert isinstance(job, Job)
    assert "ftjob" in job.id
    assert job.id in job_manager.jobs
    assert job.status == JobStatus.PENDING
    assert job.events == []
    assert job.checkpoints == []
    assert job.fine_tuned_model is None


def test_job_events(job_manager):
    job = job_manager.create_job()

    # Test logging functionality
    job.add_event(JobEvent(level="info", message="Test message"))
    assert len(job.get_events()) == 1
    event = job.get_events()[0]
    assert "ftevent" in event.id
    assert event.level == "info"
    assert event.message == "Test message"


def test_job_checkpoints(job_manager):
    job = job_manager.create_job()
    job.add_checkpoint(
        JobCheckpoint(
            fine_tuned_model_checkpoint="example_path",
            fine_tuning_job_id=job.id,
            metrics={},
            step_number=1,
        )
    )
    assert len(job.get_checkpoints()) == 1
    checkpoint = job.get_checkpoints()[0]
    assert "ftckpt" in checkpoint.id
    assert checkpoint.fine_tuned_model_checkpoint == "example_path"
    assert checkpoint.fine_tuning_job_id == job.id
    assert checkpoint.metrics == {}
    assert checkpoint.step_number == 1


def test_get_job(job_manager):
    job = job_manager.create_job()

    assert job_manager.get_job(job.id).status == JobStatus.PENDING

    with pytest.raises(ValueError):
        job_manager.get_job("nonexistent-id")


def test_get_jobs(job_manager):
    job = job_manager.create_job()
    assert len(job_manager.get_jobs()) == 1
    assert job_manager.get_jobs()[0].id == job.id

    job2 = job_manager.create_job()
    assert len(job_manager.get_jobs()) == 2
    assert job_manager.get_jobs()[1].id == job2.id
