"""
Tests for GPUManager functionality.
"""

import pytest

from arbor.server.core.config import Config
from arbor.server.services.managers.gpu_manager import GPUAllocationError, GPUManager


@pytest.fixture
def config():
    """Create a test config with 3 GPUs."""
    return Config(gpu_ids=[0, 1, 2])


@pytest.fixture
def gpu_manager(config):
    """Create a GPUManager instance for testing."""
    return GPUManager(config)


def test_gpu_manager_initialization(gpu_manager):
    """Test that GPUManager initializes with correct GPU IDs."""
    assert gpu_manager.all_gpus == {0, 1, 2}
    assert gpu_manager.gpu_allocations == {}
    assert len(gpu_manager.get_all_allocated_gpus()) == 0


def test_allocate_single_gpu(gpu_manager):
    """Test allocating a single GPU."""
    allocated = gpu_manager.allocate_gpus("job1", 1)

    assert len(allocated) == 1
    assert allocated[0] in [0, 1, 2]
    assert "job1" in gpu_manager.gpu_allocations
    assert gpu_manager.gpu_allocations["job1"] == allocated


def test_allocate_multiple_gpus(gpu_manager):
    """Test allocating multiple GPUs."""
    allocated = gpu_manager.allocate_gpus("job1", 2)

    assert len(allocated) == 2
    assert all(gpu in [0, 1, 2] for gpu in allocated)
    assert len(set(allocated)) == 2  # No duplicates
    assert "job1" in gpu_manager.gpu_allocations


def test_allocate_all_gpus(gpu_manager):
    """Test allocating all available GPUs."""
    allocated = gpu_manager.allocate_gpus("job1", 3)

    assert len(allocated) == 3
    assert set(allocated) == {0, 1, 2}
    assert gpu_manager.get_all_allocated_gpus() == {0, 1, 2}


def test_allocate_too_many_gpus(gpu_manager):
    """Test that allocating more GPUs than available raises an error."""
    with pytest.raises(GPUAllocationError) as exc_info:
        gpu_manager.allocate_gpus("job1", 4)

    assert "Not enough free GPUs" in str(exc_info.value)
    assert "Requested: 4" in str(exc_info.value)


def test_allocate_gpus_sequential(gpu_manager):
    """Test allocating GPUs to multiple jobs sequentially."""
    # Allocate 1 GPU to job1
    job1_gpus = gpu_manager.allocate_gpus("job1", 1)
    assert len(job1_gpus) == 1

    # Allocate 1 GPU to job2
    job2_gpus = gpu_manager.allocate_gpus("job2", 1)
    assert len(job2_gpus) == 1

    # Should have different GPUs
    assert set(job1_gpus).isdisjoint(set(job2_gpus))

    # Try to allocate 2 more (should fail since only 1 left)
    with pytest.raises(GPUAllocationError):
        gpu_manager.allocate_gpus("job3", 2)


def test_release_gpus(gpu_manager):
    """Test releasing GPUs from a job."""
    # Allocate GPUs
    allocated = gpu_manager.allocate_gpus("job1", 2)
    assert len(gpu_manager.get_all_allocated_gpus()) == 2

    # Release GPUs
    success = gpu_manager.release_gpus("job1")
    assert success is True
    assert len(gpu_manager.get_all_allocated_gpus()) == 0
    assert "job1" not in gpu_manager.gpu_allocations


def test_release_nonexistent_job(gpu_manager):
    """Test releasing GPUs for a job that doesn't exist."""
    success = gpu_manager.release_gpus("nonexistent_job")
    assert success is False


def test_release_and_reallocate(gpu_manager):
    """Test that released GPUs can be reallocated."""
    # Allocate all GPUs
    gpu_manager.allocate_gpus("job1", 3)

    # Try to allocate more (should fail)
    with pytest.raises(GPUAllocationError):
        gpu_manager.allocate_gpus("job2", 1)

    # Release GPUs
    gpu_manager.release_gpus("job1")

    # Now should be able to allocate
    allocated = gpu_manager.allocate_gpus("job2", 2)
    assert len(allocated) == 2


def test_get_allocated_gpus(gpu_manager):
    """Test getting GPUs allocated to a specific job."""
    allocated = gpu_manager.allocate_gpus("job1", 2)

    retrieved = gpu_manager.get_allocated_gpus("job1")
    assert retrieved == allocated

    # Test nonexistent job
    assert gpu_manager.get_allocated_gpus("nonexistent") is None


def test_get_status(gpu_manager):
    """Test getting GPU allocation status."""
    # Initially all free
    status = gpu_manager.get_status()
    assert status["total_gpus"] == [0, 1, 2]
    assert status["free_gpus"] == [0, 1, 2]
    assert status["allocated_gpus"] == []
    assert status["allocations"] == {}

    # Allocate some GPUs
    gpu_manager.allocate_gpus("job1", 2)

    status = gpu_manager.get_status()
    assert len(status["free_gpus"]) == 1
    assert len(status["allocated_gpus"]) == 2
    assert "job1" in status["allocations"]


def test_cleanup(gpu_manager):
    """Test cleanup releases all allocations."""
    # Allocate GPUs to multiple jobs
    gpu_manager.allocate_gpus("job1", 1)
    gpu_manager.allocate_gpus("job2", 1)

    assert len(gpu_manager.gpu_allocations) == 2
    assert len(gpu_manager.get_all_allocated_gpus()) == 2

    # Cleanup
    gpu_manager.cleanup()

    assert len(gpu_manager.gpu_allocations) == 0
    assert len(gpu_manager.get_all_allocated_gpus()) == 0


def test_thread_safety_simulation(gpu_manager):
    """Test that GPU allocation handles concurrent-like access."""
    # This doesn't test actual threading, but simulates the scenario
    # where multiple allocations happen in sequence

    results = []
    for i in range(3):
        try:
            allocated = gpu_manager.allocate_gpus(f"job{i}", 1)
            results.append(allocated)
        except GPUAllocationError:
            break

    # Should have allocated 3 jobs successfully
    assert len(results) == 3

    # All GPUs should be different
    all_allocated = [gpu for result in results for gpu in result]
    assert len(set(all_allocated)) == 3


def test_gpu_manager_with_empty_config():
    """Test GPUManager with no GPUs in config."""
    config = Config(gpu_ids=[])
    gpu_manager = GPUManager(config)

    assert gpu_manager.all_gpus == set()

    with pytest.raises(GPUAllocationError):
        gpu_manager.allocate_gpus("job1", 1)


def test_gpu_manager_single_gpu():
    """Test GPUManager with only one GPU."""
    config = Config(gpu_ids=[0])
    gpu_manager = GPUManager(config)

    # Can allocate the single GPU
    allocated = gpu_manager.allocate_gpus("job1", 1)
    assert allocated == [0]

    # Cannot allocate another
    with pytest.raises(GPUAllocationError):
        gpu_manager.allocate_gpus("job2", 1)

    # Release and reallocate
    gpu_manager.release_gpus("job1")
    allocated2 = gpu_manager.allocate_gpus("job2", 1)
    assert allocated2 == [0]
