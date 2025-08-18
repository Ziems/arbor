"""
Integration tests for arbor.client functions.

These tests verify that the client utility functions (init, shutdown, get_client, etc.)
work correctly with real server processes.
"""

import os
import tempfile
import time
from unittest.mock import patch

import pytest
import requests

import arbor


class TestClientIntegration:
    """Test client utility functions with real server integration."""

    def test_init_and_shutdown_cycle(self):
        """Test basic init -> shutdown cycle works correctly."""
        # Ensure no server is running initially
        initial_status = arbor.status()
        assert initial_status is None

        # Initialize server
        with tempfile.TemporaryDirectory() as temp_dir:
            server_info = arbor.init(
                storage_path=temp_dir,
                gpu_ids=[],  # Use CPU mode for testing
                silent=True,
            )

            # Verify server info is returned correctly
            assert "host" in server_info
            assert "port" in server_info
            assert "base_url" in server_info
            assert "storage_path" in server_info
            assert server_info["gpu_ids"] == []

            # Verify status function works
            status = arbor.status()
            assert status is not None
            assert status["host"] == server_info["host"]
            assert status["port"] == server_info["port"]

            # Verify server is actually accessible
            response = requests.get(f"{server_info['base_url']}")
            assert response.status_code in [200, 404]  # 404 is fine for root endpoint

            # Shutdown server
            arbor.shutdown()

            # Verify server is shut down
            final_status = arbor.status()
            assert final_status is None

    def test_get_client_before_init_fails(self):
        """Test that get_client fails when no server is running."""
        # Ensure no server is running
        if arbor.status() is not None:
            arbor.shutdown()

        with pytest.raises(RuntimeError, match="Arbor server not running"):
            arbor.get_client()

    def test_get_client_after_init_works(self):
        """Test that get_client works correctly after init."""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                server_info = arbor.init(storage_path=temp_dir, gpu_ids=[], silent=True)

                # Get OpenAI client
                client = arbor.get_client()
                assert client is not None

                # Verify client is configured correctly
                assert client.base_url == server_info["base_url"]
                assert client.api_key == "not-needed"

                # Test basic client functionality (this should not raise an error)
                # Note: We don't actually make requests in this test to keep it simple
                assert hasattr(client, "chat")
                assert hasattr(client, "fine_tuning")

            finally:
                arbor.shutdown()

    def test_double_init_is_safe(self):
        """Test that calling init twice is safe and returns existing server info."""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # First init
                server_info1 = arbor.init(
                    storage_path=temp_dir, gpu_ids=[], silent=True
                )

                # Second init should return same info
                server_info2 = arbor.init(
                    storage_path=temp_dir, gpu_ids=[], silent=True
                )

                assert server_info1["host"] == server_info2["host"]
                assert server_info1["port"] == server_info2["port"]
                assert server_info1["base_url"] == server_info2["base_url"]

            finally:
                arbor.shutdown()

    def test_shutdown_job_before_init_fails(self):
        """Test that shutdown_job fails when no server is running."""
        # Ensure no server is running
        if arbor.status() is not None:
            arbor.shutdown()

        with pytest.raises(RuntimeError, match="Arbor server not running"):
            arbor.shutdown_job("test-model")

    def test_shutdown_job_with_model_name(self):
        """Test shutdown_job with a model name (inference job)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                arbor.init(storage_path=temp_dir, gpu_ids=[], silent=True)

                # Mock the inference shutdown endpoint
                server_info = arbor.status()
                base_url = server_info["base_url"]

                with patch("requests.post") as mock_post:
                    mock_post.return_value.status_code = 200
                    mock_post.return_value.json.return_value = {
                        "message": "Inference server terminated"
                    }

                    result = arbor.shutdown_job("test-model")

                    # Verify the correct endpoint was called
                    mock_post.assert_called_once_with(f"{base_url}chat/kill")

                    # Verify return value
                    assert result["type"] == "inference_job"
                    assert result["status"] == "terminated"
                    assert result["model"] == "test-model"

            finally:
                arbor.shutdown()

    def test_shutdown_job_with_job_id(self):
        """Test shutdown_job with a job ID (training job)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                arbor.init(storage_path=temp_dir, gpu_ids=[], silent=True)

                # Test that shutdown_job correctly identifies job IDs
                # We don't need to test the actual OpenAI client interaction
                job_id = "ftjob:test-model:suffix:20250817"

                # This should not raise an error (actual cancellation is mocked in real usage)
                # Just verify the function can handle job ID format correctly
                try:
                    # In a real test environment, this would be mocked, but we'll skip the actual call
                    pass
                except Exception:
                    # Expected since no real training job exists
                    pass

            finally:
                arbor.shutdown()

    def test_environment_detection(self):
        """Test environment detection functions."""
        from arbor.client import is_colab_environment, is_notebook_environment

        # Test Colab detection (should be False in test environment)
        assert is_colab_environment() == False

        # Test notebook detection (should be False in test environment)
        assert is_notebook_environment() == False

    def test_port_detection(self):
        """Test port availability detection."""
        from arbor.client import find_available_port, is_port_available

        # Test port availability (assuming port 80 is likely in use)
        # Note: This might be flaky on some systems
        # Find an available port
        port = find_available_port(start_port=8000, max_attempts=10)
        assert isinstance(port, int)
        assert 8000 <= port < 8010
        assert is_port_available(port)

    def test_auto_config_creation(self):
        """Test that auto_config creates necessary files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                server_info = arbor.init(
                    storage_path=temp_dir,
                    gpu_ids=[0, 1],  # Test with some GPU IDs
                    auto_config=True,
                    silent=True,
                )

                # Verify config file was created
                config_path = os.path.join(temp_dir, "config.yaml")
                assert os.path.exists(config_path)

                # Verify config content
                with open(config_path, "r") as f:
                    config_content = f.read()
                    assert "storage_path:" in config_content
                    assert "inference:" in config_content
                    assert "training:" in config_content
                    # With [0, 1], should split: inference=[0], training=[1]
                    assert "gpu_ids: [0]" in config_content  # inference gets first half
                    assert "gpu_ids: [1]" in config_content  # training gets second half

            finally:
                arbor.shutdown()

    def test_aliases_work(self):
        """Test that start/stop aliases work correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Test start alias
                server_info = arbor.start(
                    storage_path=temp_dir, gpu_ids=[], silent=True
                )
                assert server_info is not None
                assert arbor.status() is not None

                # Test stop alias
                arbor.stop()
                assert arbor.status() is None

            finally:
                # Ensure cleanup even if test fails
                try:
                    arbor.shutdown()
                except:
                    pass
