"""
Tests for the new JobArtifact system and directory structure.
"""

import os
import tempfile
from pathlib import Path

import pytest

from arbor.server.api.models.schemas import GRPOInitializeRequest
from arbor.server.core.config import Config
from arbor.server.services.jobs.file_train_job import FileTrainJob
from arbor.server.services.jobs.grpo_job import GRPOJob
from arbor.server.services.jobs.inference_job import InferenceJob
from arbor.server.services.jobs.job import Job, JobArtifact


class TestJobArtifactDirectories:
    """Test that jobs create the correct directory structures."""

    @pytest.fixture
    def temp_config(self):
        """Create a temporary config for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config()
            config.storage_path = temp_dir
            yield config

    def test_base_job_default_artifacts(self, temp_config):
        """Test that base Job creates default LOGS artifact."""
        job = Job(temp_config)

        # Check that logs directory is created
        expected_log_dir = Path(temp_config.storage_path) / job.id / "logs"
        assert expected_log_dir.exists()

        # Check that log file path is set correctly
        expected_log_file = expected_log_dir / "job.log"
        assert str(Path(job.log_file_path).resolve()) == str(
            expected_log_file.resolve()
        )

    def test_job_all_artifacts(self, temp_config):
        """Test that job creates all artifact types."""
        artifacts = [
            JobArtifact.LOGS,
            JobArtifact.MODEL,
            JobArtifact.CHECKPOINTS,
            JobArtifact.METRICS,
        ]
        job = Job(temp_config, artifacts=artifacts)

        base_dir = Path(temp_config.storage_path) / job.id

        # Check all directories are created
        assert (base_dir / "logs").exists()
        assert (base_dir / "models").exists()
        assert (base_dir / "checkpoints").exists()
        assert (base_dir / "metrics").exists()

        # Check log file path
        assert str(Path(job.log_file_path).resolve()) == str(
            (base_dir / "logs" / "job.log").resolve()
        )

    def test_inference_job_structure(self, temp_config):
        """Test InferenceJob creates correct directory structure."""
        job = InferenceJob(temp_config)

        base_dir = Path(temp_config.storage_path) / job.id

        # InferenceJob should only have logs directory
        assert (base_dir / "logs").exists()
        assert not (base_dir / "models").exists()
        assert not (base_dir / "checkpoints").exists()
        assert not (base_dir / "metrics").exists()

        # Check log file path
        expected_log_file = base_dir / "logs" / "inferencejob.log"
        assert str(Path(job.log_file_path).resolve()) == str(
            expected_log_file.resolve()
        )

    def test_file_train_job_structure(self, temp_config):
        """Test FileTrainJob creates correct directory structure."""
        job = FileTrainJob(temp_config)

        base_dir = Path(temp_config.storage_path) / job.id

        # FileTrainJob should have logs, models, and checkpoints
        assert (base_dir / "logs").exists()
        assert (base_dir / "models").exists()
        assert (base_dir / "checkpoints").exists()
        assert not (base_dir / "metrics").exists()  # Should not have metrics

        # Initially uses default log file name
        expected_log_file = base_dir / "logs" / "filetrainjob.log"
        assert str(Path(job.log_file_path).resolve()) == str(
            expected_log_file.resolve()
        )

    def test_grpo_job_structure(self, temp_config):
        """Test GRPOJob creates complete directory structure."""
        # Create a minimal GRPO request
        request = GRPOInitializeRequest(
            model="test-model", train_data="test-data", suffix="test"
        )

        job = GRPOJob(temp_config, request)

        base_dir = Path(temp_config.storage_path) / job.id

        # GRPOJob should have all artifact types
        assert (base_dir / "logs").exists()
        assert (base_dir / "models").exists()
        assert (base_dir / "checkpoints").exists()
        assert (base_dir / "metrics").exists()

        # Check log file path (initially default, gets overridden later)
        expected_log_file = base_dir / "logs" / "grpojob.log"
        assert str(Path(job.log_file_path).resolve()) == str(
            expected_log_file.resolve()
        )

    def test_directory_permissions(self, temp_config):
        """Test that created directories have correct permissions."""
        job = Job(temp_config, artifacts=[JobArtifact.LOGS, JobArtifact.MODEL])

        base_dir = Path(temp_config.storage_path) / job.id
        logs_dir = base_dir / "logs"
        models_dir = base_dir / "models"

        # Check directories are readable and writable
        assert os.access(str(logs_dir), os.R_OK | os.W_OK)
        assert os.access(str(models_dir), os.R_OK | os.W_OK)

    def test_nested_directory_creation(self, temp_config):
        """Test that nested directories are created properly."""
        job = Job(temp_config, artifacts=[JobArtifact.LOGS])

        log_dir = Path(temp_config.storage_path) / job.id / "logs"
        assert log_dir.exists()
        assert log_dir.parent.exists()  # job.id directory
        assert log_dir.parent.parent.exists()  # storage_path


class TestLogCallback:
    """Test the universal log callback functionality."""

    @pytest.fixture
    def temp_config(self):
        """Create a temporary config for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config()
            config.storage_path = temp_dir
            yield config

    def test_log_callback_creates_file(self, temp_config):
        """Test that log callback creates and writes to log file in JSONL format."""
        import json

        job = Job(temp_config)

        # Create log callback and test it
        callback = job.create_log_callback("TEST")
        test_message = "This is a test log message"
        callback(test_message)

        # Check that log file was created and contains message
        assert os.path.exists(job.log_file_path)
        with open(job.log_file_path, "r") as f:
            content = f.read().strip()

            # Parse as JSONL
            log_entry = json.loads(content)

            # Verify structure
            assert log_entry["message"] == test_message
            assert log_entry["job_id"] == job.id
            assert log_entry["job_type"] == "TEST"
            assert log_entry["level"] == "info"
            assert log_entry["source"] == "training_log"
            # Check timestamp format (ISO format)
            assert "T" in log_entry["timestamp"]  # ISO timestamp contains T

    def test_log_callback_adds_events(self, temp_config):
        """Test that log callback adds events only when explicitly requested."""
        job = Job(temp_config)

        callback = job.create_log_callback("TEST")
        test_message = "Test event message"

        # By default, no event should be created
        callback(test_message)
        assert len(job.events) == 0

        # Only create event when explicitly requested
        callback(test_message, create_event=True)
        assert len(job.events) == 1
        event = job.events[0]
        assert event.message == test_message
        assert event.level == "info"
        assert event.data["source"] == "training_log"

    def test_log_callback_multiple_messages(self, temp_config):
        """Test log callback with multiple messages."""
        import json

        job = Job(temp_config)

        callback = job.create_log_callback("TEST")
        messages = ["Message 1", "Message 2", "Message 3"]

        for msg in messages:
            callback(msg)

        # Check all messages in file (JSONL format)
        with open(job.log_file_path, "r") as f:
            lines = f.read().strip().split("\n")
            assert len(lines) == 3

            for i, line in enumerate(lines):
                log_entry = json.loads(line)
                assert log_entry["message"] == messages[i]

        # No events should be created by default
        assert len(job.events) == 0

    def test_log_callback_with_extra_data(self, temp_config):
        """Test log callback with extra structured data."""
        import json

        job = Job(temp_config)

        callback = job.create_log_callback("TEST")
        test_message = "Training step completed"
        extra_data = {"step": 100, "loss": 0.25, "metrics": {"accuracy": 0.95}}

        callback(test_message, extra_data)

        # Check that log file contains extra data
        assert os.path.exists(job.log_file_path)
        with open(job.log_file_path, "r") as f:
            content = f.read().strip()
            log_entry = json.loads(content)

            # Verify base structure
            assert log_entry["message"] == test_message
            assert log_entry["job_id"] == job.id

            # Verify extra data was merged
            assert log_entry["step"] == 100
            assert log_entry["loss"] == 0.25
            assert log_entry["metrics"]["accuracy"] == 0.95

        # No event should be created by default
        assert len(job.events) == 0

        # Test with event creation
        callback(test_message, extra_data, create_event=True)
        assert len(job.events) == 1
        event = job.events[0]
        assert event.message == test_message
        assert event.data["step"] == 100
        assert event.data["loss"] == 0.25

    def test_log_callback_error_handling(self, temp_config):
        """Test log callback handles errors gracefully."""
        job = Job(temp_config)

        # Make log file path invalid to trigger error
        job.log_file_path = "/invalid/path/that/cannot/be/created.log"

        callback = job.create_log_callback("TEST")
        test_message = "This should still create an event"

        # Should not raise exception
        callback(test_message)

        # No event should be created by default (even if file write fails)
        assert len(job.events) == 0

        # Test with event creation - event should still be created even if file write fails
        callback(test_message, create_event=True)
        assert len(job.events) == 1
        assert job.events[0].message == test_message

    def test_log_structured_method(self, temp_config):
        """Test the log_structured method for direct structured logging."""
        import json

        job = Job(temp_config)

        # Test with extra data and different level
        job.log_structured(
            message="Model checkpoint saved",
            level="info",
            extra_data={
                "checkpoint_path": "/models/checkpoint-100",
                "step": 100,
                "validation_loss": 0.15,
            },
            job_type="TRAINING",
        )

        # Check that log file contains structured entry
        assert os.path.exists(job.log_file_path)
        with open(job.log_file_path, "r") as f:
            content = f.read().strip()
            log_entry = json.loads(content)

            # Verify structure
            assert log_entry["message"] == "Model checkpoint saved"
            assert log_entry["level"] == "info"
            assert log_entry["job_type"] == "TRAINING"
            assert log_entry["source"] == "direct_log"
            assert log_entry["checkpoint_path"] == "/models/checkpoint-100"
            assert log_entry["step"] == 100
            assert log_entry["validation_loss"] == 0.15

        # Check event was created (log_structured always creates events)
        assert len(job.events) == 1
        event = job.events[0]
        assert event.message == "Model checkpoint saved"
        assert event.level == "info"
        assert event.data["checkpoint_path"] == "/models/checkpoint-100"

    def test_unified_log_method(self, temp_config):
        """Test the unified log method with event control."""
        import json

        job = Job(temp_config)

        # Test logging without event creation
        job.log("Regular training progress", extra_data={"step": 50, "loss": 0.3})

        # Verify log file contains entry
        with open(job.log_file_path, "r") as f:
            content = f.read().strip()
            log_entry = json.loads(content)
            assert log_entry["message"] == "Regular training progress"
            assert log_entry["step"] == 50
            assert log_entry["loss"] == 0.3

        # No event should be created by default
        assert len(job.events) == 0

        # Test logging with event creation (for milestones)
        job.log(
            "Training completed successfully",
            level="info",
            extra_data={"final_loss": 0.15, "epochs": 10},
            job_type="TRAINING",
            create_event=True,
        )

        # Verify event was created for milestone
        assert len(job.events) == 1
        event = job.events[0]
        assert event.message == "Training completed successfully"
        assert event.level == "info"
        assert event.data["final_loss"] == 0.15
        assert event.data["epochs"] == 10


class TestJobSpecificBehavior:
    """Test job-specific directory and logging behavior."""

    @pytest.fixture
    def temp_config(self):
        """Create a temporary config for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config()
            config.storage_path = temp_dir
            yield config

    def test_file_train_job_custom_log_name(self, temp_config):
        """Test that FileTrainJob can override log file name."""
        job = FileTrainJob(temp_config)

        # Simulate what happens during fine_tune() call
        log_dir = job._make_log_dir()
        job.log_file_path = os.path.join(log_dir, "sft_training.log")

        callback = job.create_log_callback("SFT")
        callback("SFT training message")

        # Check custom log file name
        expected_path = (
            Path(temp_config.storage_path) / job.id / "logs" / "sft_training.log"
        )
        assert str(Path(job.log_file_path).resolve()) == str(expected_path.resolve())
        assert os.path.exists(job.log_file_path)

    def test_grpo_job_multiple_log_files(self, temp_config):
        """Test that GRPO job can handle multiple log files."""
        request = GRPOInitializeRequest(
            model="test-model", train_data="test-data", suffix="test"
        )
        job = GRPOJob(temp_config, request)

        # Simulate what happens during training setup
        log_dir = job._make_log_dir()
        job.log_file_path = os.path.join(log_dir, "grpo_training.log")

        # Test GRPO callback
        grpo_callback = job.create_log_callback("GRPO")
        grpo_callback("GRPO training message")

        # Check GRPO log file
        grpo_log_path = (
            Path(temp_config.storage_path) / job.id / "logs" / "grpo_training.log"
        )
        assert os.path.exists(str(grpo_log_path))

        with open(str(grpo_log_path), "r") as f:
            content = f.read()
            assert "GRPO training message" in content

    def test_job_id_uniqueness(self, temp_config):
        """Test that different jobs get unique directories."""
        job1 = Job(temp_config)
        job2 = Job(temp_config)

        # Jobs should have different IDs and directories
        assert job1.id != job2.id

        dir1 = Path(temp_config.storage_path) / job1.id
        dir2 = Path(temp_config.storage_path) / job2.id

        assert dir1 != dir2
        assert dir1.exists()
        assert dir2.exists()

    def test_directory_method_return_values(self, temp_config):
        """Test that directory methods return correct paths."""
        job = Job(temp_config)

        log_dir = job._make_log_dir()
        model_dir = job._make_model_dir()
        checkpoints_dir = job._make_checkpoints_dir()
        metrics_dir = job._make_metrics_dir()

        base_path = Path(temp_config.storage_path) / job.id

        assert str(Path(log_dir).resolve()) == str((base_path / "logs").resolve())
        assert str(Path(model_dir).resolve()) == str((base_path / "models").resolve())
        assert str(Path(checkpoints_dir).resolve()) == str(
            (base_path / "checkpoints").resolve()
        )
        assert str(Path(metrics_dir).resolve()) == str(
            (base_path / "metrics").resolve()
        )

        # All should exist after calling the methods
        assert Path(log_dir).exists()
        assert Path(model_dir).exists()
        assert Path(checkpoints_dir).exists()
        assert Path(metrics_dir).exists()
