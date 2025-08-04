"""
Pytest tests for GPU mocking functionality.

These tests verify that the GPU mocking system works correctly in minimal environments.
"""

import os
import sys
from pathlib import Path

import pytest

# Add the arbor package to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from arbor.server.utils.mock_utils import (
    get_script_path,
    get_vllm_serve_module,
    setup_mock_environment,
    should_use_mock_gpu,
)


class TestGPUMocking:
    """Test suite for GPU mocking functionality."""

    def test_mock_detection_without_env(self):
        """Test mock detection without any mock environment variables."""
        # Clear all mock-related env vars
        env_backup = {}
        for key in ["ARBOR_MOCK_GPU", "PYTEST_CURRENT_TEST", "CI", "TESTING"]:
            env_backup[key] = os.environ.pop(key, None)

        try:
            assert (
                should_use_mock_gpu() == False
            )  # Should be False when no mock env vars
        finally:
            # Restore environment
            for key, value in env_backup.items():
                if value is not None:
                    os.environ[key] = value

    def test_mock_detection_with_arbor_mock_gpu(self):
        """Test mock detection with ARBOR_MOCK_GPU environment variable."""
        os.environ["ARBOR_MOCK_GPU"] = "1"
        try:
            assert should_use_mock_gpu() == True
        finally:
            os.environ.pop("ARBOR_MOCK_GPU", None)

    def test_script_path_selection(self):
        """Test that correct script paths are selected based on mock mode."""
        script_dir = str(
            Path(__file__).parent.parent / "arbor" / "server" / "services" / "scripts"
        )

        # Test without mocking (but this will still be True due to pytest environment)
        mock_path = get_script_path("grpo_training.py", script_dir)
        assert "grpo_training_mock.py" in mock_path

        # Test with explicit mocking
        os.environ["ARBOR_MOCK_GPU"] = "1"
        try:
            mock_path = get_script_path("grpo_training.py", script_dir)
            assert "grpo_training_mock.py" in mock_path
        finally:
            os.environ.pop("ARBOR_MOCK_GPU", None)

    def test_vllm_module_selection(self):
        """Test that correct vLLM module is selected based on mock mode."""
        os.environ["ARBOR_MOCK_GPU"] = "1"
        try:
            module = get_vllm_serve_module()
            assert module == "arbor.server.services.inference.vllm_serve_mock"
        finally:
            os.environ.pop("ARBOR_MOCK_GPU", None)

    def test_environment_setup(self):
        """Test that mock environment is set up correctly."""
        base_env = {"CUDA_VISIBLE_DEVICES": "0,1,2", "SOME_OTHER_VAR": "value"}

        os.environ["ARBOR_MOCK_GPU"] = "1"
        try:
            mock_env = setup_mock_environment(base_env.copy())

            # Check that GPU mocking env vars are set
            assert mock_env.get("ARBOR_MOCK_GPU") == "1"
            assert mock_env.get("ARBOR_GPU_MOCK_MODE") == "1"
            assert mock_env.get("CUDA_VISIBLE_DEVICES") == ""

            # Check that other vars are preserved
            assert mock_env.get("SOME_OTHER_VAR") == "value"
        finally:
            os.environ.pop("ARBOR_MOCK_GPU", None)

    def test_mock_vllm_client_import(self):
        """Test that mock vLLM client can be imported and used."""
        os.environ["ARBOR_MOCK_GPU"] = "1"
        try:
            from arbor.server.services.inference.vllm_client_mock import VLLMClient

            # Create and test mock client
            client = VLLMClient("test-model", port=8890)
            assert client.model_name == "test-model"
            assert client.port == 8890

            # Test basic operations
            assert client.start_server() == True
            assert client.is_server_running() == True
            assert client.wait_for_server() == True

            # Test model info
            info = client.get_model_info()
            assert info["model_name"] == "test-model"
            assert "model_size" in info

            # Test health check
            health = client.health_check()
            assert health["status"] == "healthy"

            # Test server stats
            stats = client.get_server_stats()
            assert stats["is_running"] == True
            assert stats["model"] == "test-model"

            # Stop server
            assert client.stop_server() == True

        finally:
            os.environ.pop("ARBOR_MOCK_GPU", None)

    def test_conditional_import_in_inference_job(self):
        """Test that inference_job correctly imports mock client when mocking is enabled."""
        os.environ["ARBOR_MOCK_GPU"] = "1"
        try:
            # Clear import cache to ensure fresh import
            import sys

            if "arbor.server.services.jobs.inference_job" in sys.modules:
                del sys.modules["arbor.server.services.jobs.inference_job"]

            # Import should work with mock client
            from arbor.server.services.jobs.inference_job import VLLMClient

            # Test that it's the mock client
            client = VLLMClient("test-model")
            assert hasattr(client, "start_server")
            assert client.start_server() == True  # Mock always succeeds

        finally:
            os.environ.pop("ARBOR_MOCK_GPU", None)

    def test_mock_vllm_client_async_operations(self):
        """Test that mock vLLM client async operations work correctly."""
        import asyncio

        async def run_async_test():
            os.environ["ARBOR_MOCK_GPU"] = "1"
            try:
                from arbor.server.services.inference.vllm_client_mock import VLLMClient

                client = VLLMClient("test-model")
                client.start_server()

                # Test async generation
                messages = [{"role": "user", "content": "Hello, how are you?"}]
                response = await client.generate(messages, max_tokens=50)

                assert "choices" in response
                assert len(response["choices"]) > 0
                assert "message" in response["choices"][0]
                assert response["choices"][0]["message"]["role"] == "assistant"
                assert "content" in response["choices"][0]["message"]

                # Test chat completion
                chat_response = await client.chat_completion(messages, max_tokens=50)
                assert "choices" in chat_response

            finally:
                os.environ.pop("ARBOR_MOCK_GPU", None)

        # Run the async test
        asyncio.run(run_async_test())


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
