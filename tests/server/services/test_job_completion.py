"""
Tests for job completion monitoring and status updates.
"""

import tempfile
import threading
import time
from unittest.mock import Mock

import pytest

from arbor.server.api.models.schemas import JobStatus
from arbor.server.core.config import Config
from arbor.server.services.jobs.file_train_job import FileTrainJob


class TestJobCompletion:
    """Test job completion monitoring and status transitions."""

    @pytest.fixture
    def temp_config(self):
        """Create a temporary config for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config()
            config.storage_path = temp_dir
            yield config

    @pytest.fixture
    def mock_process_runner(self):
        """Create a mock process runner."""
        mock = Mock()
        mock.wait = Mock()
        mock.start_training = Mock()
        return mock

    @pytest.fixture
    def mock_comms_handler(self):
        """Create a mock communications handler."""
        mock = Mock()
        mock.wait_for_clients = Mock()
        return mock

    def test_successful_job_completion(
        self, temp_config, mock_process_runner, mock_comms_handler
    ):
        """Test that job status updates to SUCCEEDED on successful completion."""
        # Set up mocks
        mock_process_runner.wait.return_value = 0  # Success exit code

        job = FileTrainJob(temp_config)
        job.process_runner = mock_process_runner
        job.server_comms_handler = mock_comms_handler

        # Test the completion monitoring directly
        output_dir = "/test/output"
        job._monitor_completion(output_dir)

        # Verify job status updated correctly
        assert job.status == JobStatus.SUCCEEDED
        assert job.fine_tuned_model == f"{job.id}_model"

        # Verify process runner was called
        mock_process_runner.wait.assert_called_once()

    def test_failed_job_completion(
        self, temp_config, mock_process_runner, mock_comms_handler
    ):
        """Test that job status updates to FAILED on process failure."""
        # Set up mocks
        mock_process_runner.wait.return_value = 1  # Failure exit code

        job = FileTrainJob(temp_config)
        job.process_runner = mock_process_runner
        job.server_comms_handler = mock_comms_handler

        # Test the completion monitoring directly
        output_dir = "/test/output"
        job._monitor_completion(output_dir)

        # Verify job status updated correctly
        assert job.status == JobStatus.FAILED
        assert job.fine_tuned_model is None  # Should remain None on failure

        # Verify process runner was called
        mock_process_runner.wait.assert_called_once()

    def test_completion_monitoring_exception(
        self, temp_config, mock_process_runner, mock_comms_handler
    ):
        """Test that job status updates to FAILED when monitoring throws exception."""
        # Set up mocks
        mock_process_runner.wait.side_effect = Exception("Process monitoring failed")

        job = FileTrainJob(temp_config)
        job.process_runner = mock_process_runner
        job.server_comms_handler = mock_comms_handler

        # Test the completion monitoring directly
        output_dir = "/test/output"
        job._monitor_completion(output_dir)

        # Verify job status updated correctly on exception
        assert job.status == JobStatus.FAILED
        assert job.fine_tuned_model is None

        # Verify process runner was called
        mock_process_runner.wait.assert_called_once()

    def test_monitor_completion_method_exists(self, temp_config):
        """Test that the _monitor_completion method exists and is callable."""
        job = FileTrainJob(temp_config)

        # Verify the method exists
        assert hasattr(job, "_monitor_completion")
        assert callable(getattr(job, "_monitor_completion"))

        # Verify it takes the expected parameters (method should accept output_dir)
        import inspect

        sig = inspect.signature(job._monitor_completion)
        assert "output_dir" in sig.parameters

    def test_completion_events_created(
        self, temp_config, mock_process_runner, mock_comms_handler
    ):
        """Test that completion events are created for API access."""
        # Set up mocks
        mock_process_runner.wait.return_value = 0  # Success exit code

        job = FileTrainJob(temp_config)
        job.process_runner = mock_process_runner
        job.server_comms_handler = mock_comms_handler

        # Ensure no events initially
        assert len(job.events) == 0

        # Test the completion monitoring
        output_dir = "/test/output"
        job._monitor_completion(output_dir)

        # Verify completion event was created
        assert len(job.events) == 1
        event = job.events[0]
        assert event.message == "Training completed successfully"
        assert event.level == "info"
        assert event.data["exit_code"] == 0
        assert event.data["output_dir"] == output_dir

    def test_failure_events_created(
        self, temp_config, mock_process_runner, mock_comms_handler
    ):
        """Test that failure events are created for API access."""
        # Set up mocks
        exit_code = 42
        mock_process_runner.wait.return_value = exit_code  # Failure exit code

        job = FileTrainJob(temp_config)
        job.process_runner = mock_process_runner
        job.server_comms_handler = mock_comms_handler

        # Ensure no events initially
        assert len(job.events) == 0

        # Test the completion monitoring
        output_dir = "/test/output"
        job._monitor_completion(output_dir)

        # Verify failure event was created
        assert len(job.events) == 1
        event = job.events[0]
        assert event.message == f"Training failed with exit code {exit_code}"
        assert event.level == "error"
        assert event.data["exit_code"] == exit_code

    def test_exception_events_created(
        self, temp_config, mock_process_runner, mock_comms_handler
    ):
        """Test that exception events are created for API access."""
        # Set up mocks
        error_msg = "Process monitoring failed"
        mock_process_runner.wait.side_effect = Exception(error_msg)

        job = FileTrainJob(temp_config)
        job.process_runner = mock_process_runner
        job.server_comms_handler = mock_comms_handler

        # Ensure no events initially
        assert len(job.events) == 0

        # Test the completion monitoring
        output_dir = "/test/output"
        job._monitor_completion(output_dir)

        # Verify exception event was created
        assert len(job.events) == 1
        event = job.events[0]
        assert error_msg in event.message
        assert event.level == "error"

    def test_multiple_status_transitions(
        self, temp_config, mock_process_runner, mock_comms_handler
    ):
        """Test the complete status lifecycle: CREATED -> RUNNING -> SUCCEEDED."""
        job = FileTrainJob(temp_config)

        # Initially should be CREATED
        assert job.status == JobStatus.CREATED

        # Simulate setting to RUNNING (happens in fine_tune)
        job.status = JobStatus.RUNNING
        assert job.status == JobStatus.RUNNING

        # Mock completion monitoring
        job.process_runner = mock_process_runner
        job.server_comms_handler = mock_comms_handler
        mock_process_runner.wait.return_value = 0

        # Monitor completion
        job._monitor_completion("/test/output")

        # Should transition to SUCCEEDED
        assert job.status == JobStatus.SUCCEEDED
