"""
Utility functions for GPU mocking in subprocess environments.

This module provides functions to determine whether to use mock versions
of GPU-intensive scripts when running in test environments.
"""

import os


def should_use_mock_gpu() -> bool:
    """
    Determine if mock GPU scripts should be used.

    Returns True if:
    1. ARBOR_MOCK_GPU environment variable is set to '1', 'true', or 'yes'
    2. PYTEST_CURRENT_TEST environment variable is set (running under pytest)
    3. Any other testing indicator environment variables

    Returns:
        bool: True if mock GPU scripts should be used, False otherwise
    """
    # Check explicit mock GPU flag
    mock_gpu = os.getenv("ARBOR_MOCK_GPU", "").lower()
    if mock_gpu in ("1", "true", "yes", "on"):
        return True

    # Check if we're running under pytest
    if os.getenv("PYTEST_CURRENT_TEST") is not None:
        return True

    # Check for other common test environment indicators
    if os.getenv("CI") and os.getenv("TESTING"):
        return True

    # Check if we're in a test runner environment
    if any(
        test_var in os.environ
        for test_var in [
            "PYTEST_RUNNING",
            "UNITTEST_RUNNING",
            "TEST_MODE",
            "_PYTEST_RAISE_ON_DEPRECATED",
        ]
    ):
        return True

    return False


def get_script_path(script_name: str, script_dir: str) -> str:
    """
    Get the appropriate script path based on whether mocking is enabled.

    Args:
        script_name (str): Base script name (e.g., "grpo_training.py")
        script_dir (str): Directory containing the scripts

    Returns:
        str: Full path to the script (either real or mock version)
    """
    if should_use_mock_gpu():
        # Replace .py with _mock.py for mock version
        base_name = script_name.replace(".py", "")
        mock_script_name = f"{base_name}_mock.py"
        mock_path = os.path.join(script_dir, mock_script_name)

        # Check if mock script exists, fall back to original if not
        if os.path.exists(mock_path):
            return mock_path
        else:
            # Log warning that mock was requested but not found
            import logging

            logging.warning(
                f"Mock script {mock_path} not found, falling back to {script_name}"
            )

    # Return original script path
    return os.path.join(script_dir, script_name)


def get_vllm_serve_module() -> str:
    """
    Get the appropriate vLLM serve module name based on whether mocking is enabled.

    Returns:
        str: Module name to use with `python -m`
    """
    if should_use_mock_gpu():
        return "arbor.server.services.inference.vllm_serve_mock"
    else:
        return "arbor.server.services.inference.vllm_serve"


def setup_mock_environment(env: dict) -> dict:
    """
    Setup environment variables for mock GPU execution.

    Args:
        env (dict): Base environment dictionary

    Returns:
        dict: Environment with mock-specific variables added
    """
    if should_use_mock_gpu():
        # Set mock GPU flag for subprocess to detect
        env["ARBOR_MOCK_GPU"] = "1"

        # Override CUDA_VISIBLE_DEVICES to empty for mocking
        env["CUDA_VISIBLE_DEVICES"] = ""

        # Set other mock-related environment variables
        env["ARBOR_GPU_MOCK_MODE"] = "1"

    return env
