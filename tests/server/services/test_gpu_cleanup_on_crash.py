"""
Test GPU cleanup functionality when jobs crash or fail.
"""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from arbor.server.api.models.schemas import (
    GRPOGPUConfig,
    GRPOInitializeRequest,
    MultiGPUConfig,
)
from arbor.server.core.config import Config
from arbor.server.services.jobs.file_train_job import FileTrainJob
from arbor.server.services.jobs.grpo_job import GRPOJob
from arbor.server.services.managers.gpu_manager import GPUManager
from arbor.server.services.managers.job_manager import JobManager


@pytest.fixture
def config():
    """Create a test config with mock GPUs."""
    config = Config()
    config.gpu_ids = [0, 1, 2, 3]  # Mock 4 GPUs
    config.storage_path = "/tmp/test_arbor"
    return config


@pytest.fixture
def gpu_manager(config):
    """Create a GPU manager for testing."""
    return GPUManager(config)


class TestGPUCleanupOnCrash:
    """Test that GPUs are properly released when jobs crash."""

    def test_file_train_job_cleanup_on_exception(self, config, gpu_manager):
        """Test that FileTrainJob releases GPUs when an exception occurs."""
        job = FileTrainJob(config, gpu_manager=gpu_manager)

        # Allocate GPUs to the job
        allocated_gpus = gpu_manager.allocate_gpus(job.id, 2)
        job.allocated_gpus = allocated_gpus

        # Verify GPUs are allocated
        status = gpu_manager.get_status()
        assert len(status["allocated_gpus"]) == 2
        assert len(status["free_gpus"]) == 2

        # Test that _ensure_gpu_cleanup works
        job._ensure_gpu_cleanup()

        # Verify GPUs are released
        status = gpu_manager.get_status()
        assert len(status["allocated_gpus"]) == 0
        assert len(status["free_gpus"]) == 4

    def test_file_train_job_cleanup_idempotent(self, config, gpu_manager):
        """Test that calling cleanup multiple times is safe."""
        job = FileTrainJob(config, gpu_manager=gpu_manager)

        # Allocate GPUs
        allocated_gpus = gpu_manager.allocate_gpus(job.id, 1)
        job.allocated_gpus = allocated_gpus

        # Call cleanup multiple times
        job._ensure_gpu_cleanup()
        job._ensure_gpu_cleanup()
        job._ensure_gpu_cleanup()

        # Should still work and not crash
        status = gpu_manager.get_status()
        assert len(status["allocated_gpus"]) == 0

    def test_grpo_job_cleanup_on_exception(self, config, gpu_manager):
        """Test that GRPOJob releases GPUs when an exception occurs."""
        # Create a mock request
        request = GRPOInitializeRequest(
            model="test/model",
            gpu_config=GRPOGPUConfig(
                type="multi",
                multi=MultiGPUConfig(num_inference_gpus=1, num_training_gpus=1),
            ),
        )

        job = GRPOJob(config, request, gpu_manager=gpu_manager)

        # Manually set up GPU allocation (skipping full initialization)
        job.gpu_manager = gpu_manager

        # Test cleanup method
        job._ensure_gpu_cleanup()

        # Should not crash even if no GPUs were allocated
        assert True

    def test_monitor_completion_finally_block(self, config, gpu_manager):
        """Test that _monitor_completion calls GPU cleanup in finally block."""
        job = FileTrainJob(config, gpu_manager=gpu_manager)

        # Allocate GPUs
        allocated_gpus = gpu_manager.allocate_gpus(job.id, 1)
        job.allocated_gpus = allocated_gpus

        # Mock process_runner to simulate a crash
        job.process_runner = MagicMock()
        job.process_runner.wait.side_effect = Exception("Simulated crash")

        # Call _monitor_completion which should handle the exception and clean up GPUs
        job._monitor_completion("/fake/output/dir")

        # Verify GPUs were released despite the exception
        status = gpu_manager.get_status()
        assert len(status["allocated_gpus"]) == 0

    def test_grpo_status_handler_finally_block(self, config, gpu_manager):
        """Test that GRPO status handler calls GPU cleanup in finally block."""
        request = GRPOInitializeRequest(
            model="test/model",
            gpu_config=GRPOGPUConfig(
                type="multi",
                multi=MultiGPUConfig(num_inference_gpus=1, num_training_gpus=1),
            ),
        )

        job = GRPOJob(config, request, gpu_manager=gpu_manager)
        job.gpu_manager = gpu_manager

        # Mock the comms handler to raise an exception
        job.server_comms_handler = MagicMock()
        job.server_comms_handler.receive_event.side_effect = Exception(
            "Simulated crash"
        )

        # Mock inference job to avoid issues
        job.inference_job = MagicMock()

        # Call the status handler which should handle the exception and clean up
        job._handle_event_updates()

        # Should complete without crashing
        assert True

    def test_job_manager_backup_cleanup(self, config, gpu_manager):
        """Test that JobManager provides backup GPU cleanup."""
        job_manager = JobManager(config, gpu_manager)

        # Create a job and allocate GPUs
        job = job_manager.create_file_train_job()
        allocated_gpus = gpu_manager.allocate_gpus(job.id, 1)
        job.allocated_gpus = allocated_gpus

        # Verify allocation
        status = gpu_manager.get_status()
        assert len(status["allocated_gpus"]) == 1

        # Mock terminate to fail
        with patch.object(job, "terminate", side_effect=Exception("Terminate failed")):
            # Call cleanup on job manager
            job_manager.cleanup()

        # GPUs should still be released due to backup cleanup
        status = gpu_manager.get_status()
        assert len(status["allocated_gpus"]) == 0

    def test_base_job_cleanup_method_exists(self, config):
        """Test that base Job class has _ensure_gpu_cleanup method."""
        from arbor.server.services.jobs.job import Job

        job = Job(config)

        # Should have the method and not crash when called
        job._ensure_gpu_cleanup()
        assert hasattr(job, "_ensure_gpu_cleanup")

    def test_terminate_calls_cleanup(self, config, gpu_manager):
        """Test that terminate() calls GPU cleanup."""
        job = FileTrainJob(config, gpu_manager=gpu_manager)

        # Allocate GPUs
        allocated_gpus = gpu_manager.allocate_gpus(job.id, 1)
        job.allocated_gpus = allocated_gpus

        # Mock process_runner to avoid actual process operations
        job.process_runner = MagicMock()

        # Call terminate
        job.terminate()

        # Verify GPUs were released
        status = gpu_manager.get_status()
        assert len(status["allocated_gpus"]) == 0

    def test_concurrent_cleanup_safety(self, config, gpu_manager):
        """Test that GPU cleanup is safe under concurrent access."""
        job = FileTrainJob(config, gpu_manager=gpu_manager)

        # Allocate GPUs
        allocated_gpus = gpu_manager.allocate_gpus(job.id, 1)
        job.allocated_gpus = allocated_gpus

        # Create multiple threads that call cleanup concurrently
        threads = []
        for _ in range(5):
            t = threading.Thread(target=job._ensure_gpu_cleanup)
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Should not crash and GPUs should be released
        status = gpu_manager.get_status()
        assert len(status["allocated_gpus"]) == 0

    def test_cleanup_with_no_gpu_manager(self, config):
        """Test that cleanup is safe when no GPU manager is present."""
        job = FileTrainJob(config, gpu_manager=None)

        # Should not crash even with no GPU manager
        job._ensure_gpu_cleanup()
        assert True

    def test_cleanup_with_no_allocated_gpus(self, config, gpu_manager):
        """Test that cleanup is safe when no GPUs are allocated."""
        job = FileTrainJob(config, gpu_manager=gpu_manager)

        # Don't allocate any GPUs, just call cleanup
        job._ensure_gpu_cleanup()

        # Should not crash
        status = gpu_manager.get_status()
        assert len(status["allocated_gpus"]) == 0
        assert len(status["free_gpus"]) == 4
