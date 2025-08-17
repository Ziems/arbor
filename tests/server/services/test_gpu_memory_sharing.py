"""
Tests for GPU memory sharing functionality.
"""

import tempfile
from unittest.mock import Mock

import pytest

from arbor.server.core.config import Config
from arbor.server.services.jobs.inference_launch_config import InferenceLaunchConfig
from arbor.server.services.managers.gpu_manager import GPUManager


class TestGPUMemorySharing:
    """Test GPU memory sharing between inference and training jobs."""

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

    def test_inference_launch_config_memory_settings(self):
        """Test that InferenceLaunchConfig supports memory sharing settings."""
        # Test default settings
        config = InferenceLaunchConfig()
        assert config.gpu_memory_utilization == 0.9
        assert config.enable_gpu_sharing is False

        # Test sharing settings
        sharing_config = InferenceLaunchConfig(
            gpu_memory_utilization=0.45, enable_gpu_sharing=True
        )
        assert sharing_config.gpu_memory_utilization == 0.45
        assert sharing_config.enable_gpu_sharing is True

    def test_gpu_manager_shared_allocation(self, gpu_manager):
        """Test GPU manager's simplified allocation for sharing (same job ID gets same GPUs)."""
        # Allocate GPU to a single job that handles both inference and training
        allocated_gpus = gpu_manager.allocate_gpus("grpo_job", 1)
        assert len(allocated_gpus) == 1
        assert allocated_gpus == [0]  # Should get first available GPU

        # Check allocations in status
        status = gpu_manager.get_status()
        assert len(status["allocated_gpus"]) == 1  # Only 1 GPU physically allocated
        assert status["allocated_gpus"] == [0]
        assert "grpo_job" in status["allocations"]

        # Check that job can get its allocated GPUs
        assert gpu_manager.get_allocated_gpus("grpo_job") == [0]

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

    def test_memory_sharing_workflow(self, gpu_manager):
        """Test complete memory sharing workflow with simplified allocation."""
        # Step 1: Create inference config for sharing
        inference_config = InferenceLaunchConfig(
            gpu_ids=[0],
            gpu_memory_utilization=0.45,  # 45% for VLLM when sharing
            enable_gpu_sharing=True,
        )

        # Step 2: Allocate GPU for GRPO job (handles both inference and training)
        allocated_gpus = gpu_manager.allocate_gpus("grpo_job", 1)
        assert allocated_gpus == [0]

        # Step 3: Verify job can access the GPU
        assert gpu_manager.get_allocated_gpus("grpo_job") == [0]

        # Step 4: Check that only 1 GPU is allocated
        status = gpu_manager.get_status()
        assert len(status["allocated_gpus"]) == 1
        assert len(status["free_gpus"]) == 2  # 2 remaining GPUs still free

        # Step 5: Memory utilization verification (conceptual)
        # - VLLM inference will use 45% of GPU 0 memory (via --gpu-memory-utilization)
        # - Training will use remaining ~50% of GPU 0 memory (PyTorch auto-allocates)
        # - Both processes run under the same job ID but in separate subprocesses
        assert inference_config.gpu_memory_utilization == 0.45
        assert inference_config.enable_gpu_sharing is True

    def test_structured_gpu_config_single_shared(self, gpu_manager):
        """Test new structured GPU configuration for single GPU with sharing."""
        from arbor.server.api.models.schemas import GRPOGPUConfig, SingleGPUConfig

        # Test single GPU with shared memory config
        gpu_config = GRPOGPUConfig(
            type="single", single=SingleGPUConfig(shared_memory=True)
        )

        # Verify the config structure
        assert gpu_config.type == "single"
        assert gpu_config.single.shared_memory is True
        assert gpu_config.multi is None

    def test_structured_gpu_config_multi(self, gpu_manager):
        """Test new structured GPU configuration for multi-GPU setup."""
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
        assert gpu_config.single is None
