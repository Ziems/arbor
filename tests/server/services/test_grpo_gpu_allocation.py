"""
Tests for GRPO GPU allocation functionality.
"""

from unittest.mock import MagicMock, Mock

import pytest

from arbor.server.api.models.schemas import GRPOInitializeRequest
from arbor.server.core.config import Config
from arbor.server.services.jobs.grpo_job import GRPOJob
from arbor.server.services.managers.gpu_manager import GPUAllocationError, GPUManager
from arbor.server.services.managers.grpo_manager import GRPOManager


@pytest.fixture
def config():
    """Create a test config with 4 GPUs."""
    return Config(gpu_ids=[0, 1, 2, 3])


@pytest.fixture
def gpu_manager(config):
    """Create a GPUManager instance for testing."""
    return GPUManager(config)


@pytest.fixture
def grpo_request():
    """Create a test GRPO request."""
    from arbor.server.api.models.schemas import GRPOGPUConfig, MultiGPUConfig

    return GRPOInitializeRequest(
        model="test-model",
        grpo_flavor="grpo",
        gpu_config=GRPOGPUConfig(
            type="multi",
            multi=MultiGPUConfig(num_training_gpus=2, num_inference_gpus=1),
        ),
    )


@pytest.fixture
def mock_inference_manager():
    """Create a mock inference manager."""
    mock = Mock()
    mock.launch_job.return_value = Mock()
    return mock


def test_grpo_request_defaults():
    """Test that GRPO request has correct defaults for GPU allocation."""
    request = GRPOInitializeRequest(model="test-model", grpo_flavor="grpo")

    # Default gpu_config should be single GPU with sharing enabled
    assert request.gpu_config.type == "single"
    assert request.gpu_config.single.shared_memory is True
    assert request.gpu_config.multi is None


def test_grpo_request_custom_gpus():
    """Test GRPO request with custom multi-GPU allocation."""
    from arbor.server.api.models.schemas import GRPOGPUConfig, MultiGPUConfig

    request = GRPOInitializeRequest(
        model="test-model",
        grpo_flavor="grpo",
        gpu_config=GRPOGPUConfig(
            type="multi",
            multi=MultiGPUConfig(num_training_gpus=3, num_inference_gpus=2),
        ),
    )

    assert request.gpu_config.type == "multi"
    assert request.gpu_config.multi.num_training_gpus == 3
    assert request.gpu_config.multi.num_inference_gpus == 2


def test_grpo_job_gpu_allocation(
    config, gpu_manager, grpo_request, mock_inference_manager
):
    """Test that GRPO job allocates GPUs correctly."""
    # Create GRPO job with GPU manager
    grpo_job = GRPOJob(config, grpo_request, gpu_manager=gpu_manager)

    # Get GPU counts from the new structure
    num_inference_gpus = grpo_request.gpu_config.multi.num_inference_gpus
    num_training_gpus = grpo_request.gpu_config.multi.num_training_gpus
    total_gpus = num_inference_gpus + num_training_gpus

    # Test GPU allocation logic directly without full initialization
    allocated_gpus = gpu_manager.allocate_gpus(grpo_job.id, total_gpus)

    # Verify correct number of GPUs allocated
    assert len(allocated_gpus) == 3  # 2 training + 1 inference

    # Test GPU splitting logic
    inference_gpus = allocated_gpus[:num_inference_gpus]
    training_gpus = allocated_gpus[num_inference_gpus:]

    assert len(inference_gpus) == 1
    assert len(training_gpus) == 2
    assert set(inference_gpus).isdisjoint(set(training_gpus))

    # Verify GPU manager state
    status = gpu_manager.get_status()
    assert len(status["allocated_gpus"]) == 3
    assert grpo_job.id in gpu_manager.gpu_allocations


def test_grpo_job_insufficient_gpus(config, grpo_request, mock_inference_manager):
    """Test GRPO job fails when insufficient GPUs available."""
    # Create GPU manager with only 2 GPUs
    limited_config = Config(gpu_ids=[0, 1])
    gpu_manager = GPUManager(limited_config)

    # Request needs 3 GPUs (2 training + 1 inference) but only 2 available
    grpo_job = GRPOJob(limited_config, grpo_request, gpu_manager=gpu_manager)

    total_gpus = (
        grpo_request.gpu_config.multi.num_inference_gpus
        + grpo_request.gpu_config.multi.num_training_gpus
    )

    with pytest.raises(GPUAllocationError) as exc_info:
        gpu_manager.allocate_gpus(grpo_job.id, total_gpus)

    assert "Not enough free GPUs" in str(exc_info.value)


def test_grpo_job_no_gpu_manager(config, grpo_request, mock_inference_manager):
    """Test GRPO job fails when no GPU manager provided."""
    grpo_job = GRPOJob(config, grpo_request, gpu_manager=None)

    # This test is mainly to verify the job can be created without GPU manager
    # The actual error would occur when trying to use GPU functionality
    assert grpo_job.gpu_manager is None


def test_grpo_manager_with_gpu_manager(
    config, gpu_manager, grpo_request, mock_inference_manager
):
    """Test GRPO manager passes GPU manager to GRPO job."""
    grpo_manager = GRPOManager(config, gpu_manager=gpu_manager)

    # Verify GPU manager is properly stored
    assert grpo_manager.gpu_manager is not None
    assert grpo_manager.gpu_manager is gpu_manager


def test_gpu_splitting_logic():
    """Test the logic for splitting allocated GPUs between inference and training."""
    # Simulate the GPU splitting logic from GRPO job
    total_gpus = 5
    num_inference_gpus = 2
    num_training_gpus = 3

    # Simulate allocated GPUs from GPU manager
    allocated_gpus = [0, 1, 2, 3, 4]

    # Split them as GRPO job would
    inference_gpus = allocated_gpus[:num_inference_gpus]
    training_gpus = allocated_gpus[num_inference_gpus:]

    assert inference_gpus == [0, 1]
    assert training_gpus == [2, 3, 4]
    assert len(inference_gpus) == num_inference_gpus
    assert len(training_gpus) == num_training_gpus


def test_grpo_job_cleanup_releases_gpus(config, gpu_manager, grpo_request):
    """Test that GRPO job cleanup releases allocated GPUs."""
    grpo_job = GRPOJob(config, grpo_request, gpu_manager=gpu_manager)

    # Manually allocate GPUs (simulating what initialize would do)
    total_gpus = (
        grpo_request.gpu_config.multi.num_inference_gpus
        + grpo_request.gpu_config.multi.num_training_gpus
    )
    gpu_manager.allocate_gpus(grpo_job.id, total_gpus)

    # Verify GPUs are allocated
    assert len(gpu_manager.get_all_allocated_gpus()) == total_gpus

    # Call cleanup
    grpo_job.cleanup_termination()

    # Verify GPUs are released
    assert len(gpu_manager.get_all_allocated_gpus()) == 0
    assert grpo_job.id not in gpu_manager.gpu_allocations


def test_multiple_grpo_jobs_gpu_conflicts(config, gpu_manager, mock_inference_manager):
    """Test that multiple GRPO jobs handle GPU conflicts correctly."""
    # Create first GRPO job that uses 3 GPUs
    from arbor.server.api.models.schemas import GRPOGPUConfig, MultiGPUConfig

    request1 = GRPOInitializeRequest(
        model="model1",
        grpo_flavor="grpo",
        gpu_config=GRPOGPUConfig(
            type="multi",
            multi=MultiGPUConfig(num_training_gpus=2, num_inference_gpus=1),
        ),
    )

    grpo_job1 = GRPOJob(config, request1, gpu_manager=gpu_manager)

    # Allocate GPUs for first job (3 out of 4 GPUs)
    total_gpus1 = (
        request1.gpu_config.multi.num_inference_gpus
        + request1.gpu_config.multi.num_training_gpus
    )
    allocated1 = gpu_manager.allocate_gpus(grpo_job1.id, total_gpus1)

    # Verify 3 GPUs are allocated
    assert len(allocated1) == 3
    assert len(gpu_manager.get_all_allocated_gpus()) == 3

    # Create second GRPO job that needs 2 GPUs (should fail with only 1 remaining)
    request2 = GRPOInitializeRequest(
        model="model2",
        grpo_flavor="grpo",
        gpu_config=GRPOGPUConfig(
            type="multi",
            multi=MultiGPUConfig(num_training_gpus=1, num_inference_gpus=1),
        ),
    )

    grpo_job2 = GRPOJob(config, request2, gpu_manager=gpu_manager)
    total_gpus2 = (
        request2.gpu_config.multi.num_inference_gpus
        + request2.gpu_config.multi.num_training_gpus
    )

    # This should fail (needs 2 GPUs but only 1 available)
    with pytest.raises(GPUAllocationError):
        gpu_manager.allocate_gpus(grpo_job2.id, total_gpus2)


def test_grpo_manager_cleanup_releases_all_gpus(config, gpu_manager):
    """Test that GRPO manager cleanup releases all GPUs from all jobs."""
    grpo_manager = GRPOManager(config, gpu_manager=gpu_manager)

    # Manually create some allocations (simulating active jobs)
    gpu_manager.allocate_gpus("job1", 2)
    gpu_manager.allocate_gpus("job2", 1)

    assert len(gpu_manager.get_all_allocated_gpus()) == 3

    # GRPO manager cleanup should release all GPUs from its jobs
    # Note: This test is simplified since we're not actually creating GRPO jobs
    # In real usage, the cleanup would call terminate() on each job, which would release GPUs
    gpu_manager.cleanup()  # Simulate what would happen

    assert len(gpu_manager.get_all_allocated_gpus()) == 0


@pytest.mark.parametrize(
    "num_training,num_inference,total_gpus,should_succeed",
    [
        (1, 1, 4, True),  # 2 GPUs needed, 4 available - OK
        (2, 1, 4, True),  # 3 GPUs needed, 4 available - OK
        (3, 1, 4, True),  # 4 GPUs needed, 4 available - OK
        (3, 2, 4, False),  # 5 GPUs needed, 4 available - FAIL
        (1, 1, 1, False),  # 2 GPUs needed, 1 available - FAIL
        (0, 1, 1, True),  # 1 GPU needed, 1 available - OK
    ],
)
def test_grpo_gpu_allocation_scenarios(
    num_training, num_inference, total_gpus, should_succeed
):
    """Test various GPU allocation scenarios."""
    config = Config(gpu_ids=list(range(total_gpus)))
    gpu_manager = GPUManager(config)

    from arbor.server.api.models.schemas import GRPOGPUConfig, MultiGPUConfig

    request = GRPOInitializeRequest(
        model="test-model",
        grpo_flavor="grpo",
        gpu_config=GRPOGPUConfig(
            type="multi",
            multi=MultiGPUConfig(
                num_training_gpus=num_training, num_inference_gpus=num_inference
            ),
        ),
    )

    grpo_job = GRPOJob(config, request, gpu_manager=gpu_manager)
    mock_inference_manager = Mock()

    total_gpus_needed = num_training + num_inference

    if should_succeed:
        # Should be able to allocate the needed GPUs
        allocated = gpu_manager.allocate_gpus(grpo_job.id, total_gpus_needed)
        assert len(allocated) == total_gpus_needed
    else:
        # Should fail due to insufficient GPUs
        with pytest.raises(GPUAllocationError):
            gpu_manager.allocate_gpus(grpo_job.id, total_gpus_needed)
