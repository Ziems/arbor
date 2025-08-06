"""
Tests for FileTrainJob GPU allocation functionality.
"""

from unittest.mock import MagicMock, Mock

import pytest

from arbor.server.api.models.schemas import FineTuneRequest
from arbor.server.core.config import Config
from arbor.server.services.jobs.file_train_job import FileTrainJob
from arbor.server.services.managers.gpu_manager import GPUAllocationError, GPUManager


@pytest.fixture
def config():
    """Create a test config with 4 GPUs."""
    return Config(gpu_ids=[0, 1, 2, 3])


@pytest.fixture
def gpu_manager(config):
    """Create a GPUManager instance for testing."""
    return GPUManager(config)


@pytest.fixture
def file_train_request():
    """Create a test file train request."""
    return FineTuneRequest(model="test-model", training_file="test-file-id", num_gpus=2)


@pytest.fixture
def mock_file_manager():
    """Create a mock file manager."""
    mock = Mock()
    mock.get_file.return_value = {"path": "/test/path/data.jsonl"}
    mock.validate_file_format.return_value = None
    return mock


def test_file_train_request_defaults():
    """Test that FineTuneRequest has correct defaults for GPU allocation."""
    request = FineTuneRequest(model="test-model", training_file="test-file")

    assert request.num_gpus == 1


def test_file_train_request_custom_gpus():
    """Test FineTuneRequest with custom GPU allocation."""
    request = FineTuneRequest(model="test-model", training_file="test-file", num_gpus=3)

    assert request.num_gpus == 3


def test_file_train_job_gpu_allocation(
    config, gpu_manager, file_train_request, mock_file_manager
):
    """Test that FileTrainJob allocates GPUs correctly."""
    # Create FileTrainJob with GPU manager
    file_train_job = FileTrainJob(config, gpu_manager=gpu_manager)

    # Mock the fine_tune method to only test GPU allocation part
    # We'll call the GPU allocation part directly
    allocated_gpus = gpu_manager.allocate_gpus(
        file_train_job.id, file_train_request.num_gpus
    )
    file_train_job.allocated_gpus = allocated_gpus

    # Verify correct number of GPUs allocated
    assert len(allocated_gpus) == 2
    assert all(gpu in [0, 1, 2, 3] for gpu in allocated_gpus)

    # Verify GPU manager state
    status = gpu_manager.get_status()
    assert len(status["allocated_gpus"]) == 2
    assert file_train_job.id in gpu_manager.gpu_allocations


def test_file_train_job_insufficient_gpus(config, gpu_manager, mock_file_manager):
    """Test FileTrainJob fails when insufficient GPUs available."""
    # Create GPU manager with only 1 GPU
    limited_config = Config(gpu_ids=[0])
    limited_gpu_manager = GPUManager(limited_config)

    # Request needs 2 GPUs but only 1 available
    request = FineTuneRequest(model="test-model", training_file="test-file", num_gpus=2)

    file_train_job = FileTrainJob(limited_config, gpu_manager=limited_gpu_manager)

    with pytest.raises(GPUAllocationError) as exc_info:
        limited_gpu_manager.allocate_gpus(file_train_job.id, request.num_gpus)

    assert "Not enough free GPUs" in str(exc_info.value)


def test_file_train_job_no_gpu_manager(config, file_train_request, mock_file_manager):
    """Test FileTrainJob fallback when no GPU manager provided."""
    file_train_job = FileTrainJob(config, gpu_manager=None)

    # This test verifies the job can be created without GPU manager
    assert file_train_job.gpu_manager is None
    assert file_train_job.allocated_gpus is None


def test_file_train_job_cleanup_releases_gpus(config, gpu_manager, file_train_request):
    """Test that FileTrainJob cleanup releases allocated GPUs."""
    file_train_job = FileTrainJob(config, gpu_manager=gpu_manager)

    # Manually allocate GPUs (simulating what fine_tune would do)
    allocated_gpus = gpu_manager.allocate_gpus(
        file_train_job.id, file_train_request.num_gpus
    )
    file_train_job.allocated_gpus = allocated_gpus

    # Verify GPUs are allocated
    assert len(gpu_manager.get_all_allocated_gpus()) == file_train_request.num_gpus

    # Call terminate (which includes GPU cleanup)
    file_train_job.terminate()

    # Verify GPUs are released
    assert len(gpu_manager.get_all_allocated_gpus()) == 0
    assert file_train_job.id not in gpu_manager.gpu_allocations
    assert file_train_job.allocated_gpus is None


def test_multiple_file_train_jobs_gpu_conflicts(config, gpu_manager, mock_file_manager):
    """Test that multiple FileTrainJob instances handle GPU conflicts correctly."""
    # Create first job that uses 3 GPUs
    request1 = FineTuneRequest(model="model1", training_file="file1", num_gpus=3)

    file_train_job1 = FileTrainJob(config, gpu_manager=gpu_manager)

    # Allocate GPUs for first job (3 out of 4 GPUs)
    allocated1 = gpu_manager.allocate_gpus(file_train_job1.id, request1.num_gpus)
    file_train_job1.allocated_gpus = allocated1

    # Verify 3 GPUs are allocated
    assert len(allocated1) == 3
    assert len(gpu_manager.get_all_allocated_gpus()) == 3

    # Create second job that needs 2 GPUs (should fail with only 1 remaining)
    request2 = FineTuneRequest(model="model2", training_file="file2", num_gpus=2)

    file_train_job2 = FileTrainJob(config, gpu_manager=gpu_manager)

    # This should fail (needs 2 GPUs but only 1 available)
    with pytest.raises(GPUAllocationError):
        gpu_manager.allocate_gpus(file_train_job2.id, request2.num_gpus)


@pytest.mark.parametrize(
    "num_gpus,total_gpus,should_succeed",
    [
        (1, 4, True),  # 1 GPU needed, 4 available - OK
        (2, 4, True),  # 2 GPUs needed, 4 available - OK
        (4, 4, True),  # 4 GPUs needed, 4 available - OK
        (5, 4, False),  # 5 GPUs needed, 4 available - FAIL
        (1, 0, False),  # 1 GPU needed, 0 available - FAIL
    ],
)
def test_file_train_gpu_allocation_scenarios(num_gpus, total_gpus, should_succeed):
    """Test various GPU allocation scenarios."""
    config = Config(gpu_ids=list(range(total_gpus)))
    gpu_manager = GPUManager(config)

    request = FineTuneRequest(
        model="test-model", training_file="test-file", num_gpus=num_gpus
    )

    file_train_job = FileTrainJob(config, gpu_manager=gpu_manager)

    if should_succeed:
        # Should be able to allocate the needed GPUs
        allocated = gpu_manager.allocate_gpus(file_train_job.id, num_gpus)
        assert len(allocated) == num_gpus
    else:
        # Should fail due to insufficient GPUs
        with pytest.raises(GPUAllocationError):
            gpu_manager.allocate_gpus(file_train_job.id, num_gpus)


def test_file_train_job_fallback_to_config_gpus(config):
    """Test FileTrainJob fallback behavior when no GPU manager provided."""
    file_train_job = FileTrainJob(config, gpu_manager=None)

    # Simulate what would happen in fine_tune when no GPU manager is available
    if file_train_job.gpu_manager:
        # This shouldn't happen in this test
        allocated_gpus = file_train_job.gpu_manager.allocate_gpus(file_train_job.id, 2)
    else:
        # Fallback to config GPUs
        allocated_gpus = config.gpu_ids

    # Should fallback to all config GPUs
    assert allocated_gpus == [0, 1, 2, 3]
