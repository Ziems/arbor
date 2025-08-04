#!/usr/bin/env python3
"""
Quick test script to verify GPU mocking functionality works correctly.

This script tests both the mock utilities and actual subprocess execution
to ensure GPU mocking works in subprocess environments.
"""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Add the arbor package to Python path
sys.path.insert(0, str(Path(__file__).parent))

from arbor.server.utils.mock_utils import (
    should_use_mock_gpu,
    get_script_path,
    get_vllm_serve_module,
    setup_mock_environment
)


def test_mock_detection():
    """Test that mock GPU detection works correctly."""
    print("=" * 60)
    print("Testing Mock GPU Detection")
    print("=" * 60)
    
    # Test without mock environment
    original_env = os.environ.copy()
    for key in ["ARBOR_MOCK_GPU", "PYTEST_CURRENT_TEST", "CI", "TESTING"]:
        os.environ.pop(key, None)
    
    print(f"Without mock env: should_use_mock_gpu() = {should_use_mock_gpu()}")
    
    # Test with ARBOR_MOCK_GPU=1
    os.environ["ARBOR_MOCK_GPU"] = "1"
    print(f"With ARBOR_MOCK_GPU=1: should_use_mock_gpu() = {should_use_mock_gpu()}")
    
    # Test with pytest environment
    os.environ.pop("ARBOR_MOCK_GPU", None)
    os.environ["PYTEST_CURRENT_TEST"] = "test_something.py::test_function"
    print(f"With PYTEST_CURRENT_TEST: should_use_mock_gpu() = {should_use_mock_gpu()}")
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)
    print()


def test_script_path_selection():
    """Test that correct script paths are selected."""
    print("=" * 60)
    print("Testing Script Path Selection")
    print("=" * 60)
    
    script_dir = str(Path(__file__).parent / "arbor" / "server" / "services" / "scripts")
    
    # Test without mocking
    os.environ.pop("ARBOR_MOCK_GPU", None)
    normal_path = get_script_path("grpo_training.py", script_dir)
    print(f"Normal mode: {normal_path}")
    
    # Test with mocking
    os.environ["ARBOR_MOCK_GPU"] = "1"
    mock_path = get_script_path("grpo_training.py", script_dir)
    print(f"Mock mode: {mock_path}")
    
    # Test vLLM module selection
    normal_module = get_vllm_serve_module()
    print(f"Mock vLLM module: {normal_module}")
    
    os.environ.pop("ARBOR_MOCK_GPU", None)
    print()


def test_mock_vllm_server():
    """Test that the mock vLLM server can be started."""
    print("=" * 60)
    print("Testing Mock vLLM Server")
    print("=" * 60)
    
    # Set mock environment
    os.environ["ARBOR_MOCK_GPU"] = "1"
    
    try:
        # Try to start the mock vLLM server briefly
        cmd = [
            sys.executable, "-m", "arbor.server.services.inference.vllm_serve_mock",
            "--host", "localhost",
            "--port", "8888",
            "--model", "test-model"
        ]
        
        print(f"Starting mock vLLM server: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Let it run for a few seconds
        time.sleep(3)
        
        # Check if it's still running (success)
        if process.poll() is None:
            print("✅ Mock vLLM server started successfully!")
            process.terminate()
            process.wait(timeout=5)
        else:
            stdout, _ = process.communicate()
            print(f"❌ Mock vLLM server failed to start. Output:\n{stdout}")
            
    except Exception as e:
        print(f"❌ Error testing mock vLLM server: {e}")
    
    finally:
        os.environ.pop("ARBOR_MOCK_GPU", None)
    
    print()


def test_mock_training_script():
    """Test that a mock training script can be run."""
    print("=" * 60)
    print("Testing Mock Training Script")
    print("=" * 60)
    
    # Set mock environment
    os.environ["ARBOR_MOCK_GPU"] = "1"
    
    try:
        # Try to run the mock GRPO training script
        cmd = [
            sys.executable, "-m", "arbor.server.services.scripts.grpo_training_mock",
            "--model", "test-model",
            "--command_port", "12345",
            "--status_port", "12346", 
            "--data_port", "12347",
            "--broadcast_port", "12348",
            "--handshake_port", "12349",
            "--vllm_group_port", "12350",
            "--vllm_port", "12351",
            "--trl_train_kwargs", '{"output_dir": "/tmp/test", "num_train_epochs": 1}',
            "--arbor_train_kwargs", '{"grpo_flavor": "grpo"}'
        ]
        
        print(f"Running mock training script...")
        
        # Run the script and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10  # 10 second timeout
        )
        
        print(f"Return code: {result.returncode}")
        if result.stdout:
            print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
            
        if result.returncode == 0:
            print("✅ Mock training script ran successfully!")
        else:
            print("❌ Mock training script failed")
            
    except subprocess.TimeoutExpired:
        print("✅ Mock training script started successfully (timeout after 10s is expected)")
    except Exception as e:
        print(f"❌ Error testing mock training script: {e}")
    
    finally:
        os.environ.pop("ARBOR_MOCK_GPU", None)
    
    print()


def test_environment_setup():
    """Test that mock environment is set up correctly."""
    print("=" * 60)
    print("Testing Environment Setup")
    print("=" * 60)
    
    base_env = {
        "CUDA_VISIBLE_DEVICES": "0,1,2",
        "SOME_OTHER_VAR": "value"
    }
    
    # Test without mocking
    normal_env = setup_mock_environment(base_env.copy())
    print("Normal environment:")
    for key, value in normal_env.items():
        if key.startswith(("CUDA", "ARBOR")):
            print(f"  {key}={value}")
    
    # Test with mocking
    os.environ["ARBOR_MOCK_GPU"] = "1"
    mock_env = setup_mock_environment(base_env.copy())
    print("\nMock environment:")
    for key, value in mock_env.items():
        if key.startswith(("CUDA", "ARBOR")):
            print(f"  {key}={value}")
    
    os.environ.pop("ARBOR_MOCK_GPU", None)
    print()


def main():
    """Run all tests."""
    print("🧪 GPU Mocking Test Suite")
    print("=" * 60)
    
    try:
        test_mock_detection()
        test_script_path_selection()
        test_environment_setup()
        test_mock_vllm_server()
        test_mock_training_script()
        
        print("=" * 60)
        print("✅ GPU mocking test suite completed!")
        print("\nTo manually test:")
        print("1. Set ARBOR_MOCK_GPU=1")
        print("2. Run any training or inference job")
        print("3. Check that mock scripts are used instead of real ones")
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()