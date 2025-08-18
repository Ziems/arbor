"""
Tests for basic GPU allocation functionality between inference and training jobs.

This tests the standard multi-GPU allocation system, not memory sharing.
"""

import tempfile
from unittest.mock import Mock

import pytest

from arbor.server.core.config import Config
from arbor.server.services.jobs.inference_launch_config import InferenceLaunchConfig
from arbor.server.services.managers.gpu_manager import GPUManager


class TestGPUAllocation:
    """Test basic GPU allocation between inference and training jobs."""

    @pytest.fixture
    def temp_config(self):
        """Create a temporary config for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config()
            config.storage_path = temp_dir
            config.gpu_ids = [0, 1, 2]  # 3 GPUs available
            yield config

    @pytest.fixture
    def gpu_manager(self, temp_config):
        """Create a GPU manager for testing."""
        return GPUManager(temp_config)

    def test_inference_launch_config_basic(self):
        """Test that InferenceLaunchConfig has basic functionality."""
        # Test default settings
        config = InferenceLaunchConfig()
        assert config.max_context_length is None
        assert config.gpu_ids is None
        assert config.is_grpo is False
        assert config.grpo_job_id is None

        # Test with custom values
        custom_config = InferenceLaunchConfig(
            max_context_length=4096,
            gpu_ids=[0, 1],
            is_grpo=True,
            grpo_job_id="test-job",
        )
        assert custom_config.max_context_length == 4096
        assert custom_config.gpu_ids == [0, 1]
        assert custom_config.is_grpo is True
        assert custom_config.grpo_job_id == "test-job"

    def test_gpu_manager_basic_allocation(self, gpu_manager):
        """Test GPU manager's basic allocation functionality."""
        # Allocate GPUs to different jobs
        allocated_gpus1 = gpu_manager.allocate_gpus("job1", 1)
        assert len(allocated_gpus1) == 1
        assert allocated_gpus1 == [0]  # Should get first available GPU

        allocated_gpus2 = gpu_manager.allocate_gpus("job2", 1)
        assert len(allocated_gpus2) == 1
        assert allocated_gpus2 == [1]  # Should get second available GPU

        # Check allocations in status
        status = gpu_manager.get_status()
        assert len(status["allocated_gpus"]) == 2
        assert status["allocated_gpus"] == [0, 1]
        assert "job1" in status["allocations"]
        assert "job2" in status["allocations"]

    def test_gpu_manager_allocation_error(self, gpu_manager):
        """Test error handling when not enough GPUs are available."""
        # Allocate all available GPUs (3 GPUs in test config)
        gpu_manager.allocate_gpus("job1", 3)

        # Try to allocate more GPUs than available
        with pytest.raises(Exception) as excinfo:
            gpu_manager.allocate_gpus("job2", 1)

        assert "Not enough free GPUs" in str(excinfo.value)

    def test_gpu_manager_release_gpus(self, gpu_manager):
        """Test releasing GPU allocations."""
        # Allocate GPUs to jobs
        gpu_manager.allocate_gpus("job1", 1)
        gpu_manager.allocate_gpus("job2", 1)

        # Release job1
        result = gpu_manager.release_gpus("job1")
        assert result is True

        # Check that job2 still has its allocation
        assert gpu_manager.get_allocated_gpus("job2") == [1]
        assert gpu_manager.get_allocated_gpus("job1") is None

        # Check status
        status = gpu_manager.get_status()
        assert "job1" not in status["allocations"]
        assert "job2" in status["allocations"]

    def test_gpu_manager_cleanup(self, gpu_manager):
        """Test that cleanup clears all allocations."""
        # Set up allocations
        gpu_manager.allocate_gpus("job1", 1)
        gpu_manager.allocate_gpus("job2", 1)

        # Verify allocations exist
        status = gpu_manager.get_status()
        assert len(status["allocations"]) == 2

        # Cleanup
        gpu_manager.cleanup()

        # Verify all allocations are cleared
        status = gpu_manager.get_status()
        assert len(status["allocations"]) == 0
        assert len(status["allocated_gpus"]) == 0

    def test_structured_gpu_config_multi(self):
        """Test structured GPU configuration for multi-GPU setup."""
        from arbor.server.api.models.schemas import GRPOGPUConfig, MultiGPUConfig

        # Test multi-GPU config
        gpu_config = GRPOGPUConfig(
            type="multi",
            multi=MultiGPUConfig(num_inference_gpus=2, num_training_gpus=1),
        )

        # Verify the config structure
        assert gpu_config.type == "multi"
        assert gpu_config.multi.num_inference_gpus == 2
        assert gpu_config.multi.num_training_gpus == 1
