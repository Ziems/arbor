# Arbor Development Guide

This guide contains information for developers working on the Arbor codebase.

## Table of Contents

- [Development Environment Setup](#development-environment-setup)
- [GPU Mocking for Testing](#gpu-mocking-for-testing)
- [Testing](#testing)
- [Code Style](#code-style)

## Development Environment Setup

### Minimal Development Environment

For development work that doesn't require actual GPU operations (testing, API development, etc.), you can use the minimal dependency group:

```bash
# Install minimal dependencies (no heavy ML libraries)
uv sync --only-group minimal

# Activate the environment
source .venv/bin/activate
```

This installs only the core dependencies needed for development:
- FastAPI, uvicorn (API server)
- Click (CLI)
- Pytest, black, isort (development tools)
- Basic utilities (httpx, pyyaml, coolname, etc.)

### Full Environment

For full development including GPU operations:

```bash
# Install all dependencies (requires cmake, pkg-config for building)
uv sync --dev

# Activate the environment
source .venv/bin/activate
```

## GPU Mocking for Testing

Arbor includes a comprehensive GPU mocking system that allows you to test training and inference functionality without requiring actual GPU hardware or heavy ML dependencies.

### Overview

The GPU mocking system works by:
1. **Detecting test environments** automatically
2. **Substituting mock scripts** for GPU-intensive operations in subprocess environments
3. **Simulating GPU operations** without importing PyTorch, vLLM, or other heavy libraries
4. **Maintaining identical CLI interfaces** between real and mock versions

### Quick Start

```bash
# Enable GPU mocking
export ARBOR_MOCK_GPU=1

# Now all GPU operations will use lightweight mocks
python -m arbor.server.services.scripts.grpo_training_mock --help
```

### Automatic Detection

GPU mocking is automatically enabled when:
- `ARBOR_MOCK_GPU=1` environment variable is set
- Running under pytest (`PYTEST_CURRENT_TEST` is set)
- Running in CI environments with `CI=1` and `TESTING=1`
- Other common test environment indicators are present

### Mock Scripts

The following mock scripts are available:

| Real Script | Mock Script | Purpose |
|-------------|-------------|---------|
| `vllm_serve.py` | `vllm_serve_mock.py` | Mock vLLM inference server |
| `grpo_training.py` | `grpo_training_mock.py` | Mock GRPO training |
| `mmgrpo_training.py` | `mmgrpo_training_mock.py` | Mock multi-modal GRPO |
| `sft_training.py` | `sft_training_mock.py` | Mock supervised fine-tuning |
| `dpo_training.py` | `dpo_training_mock.py` | Mock DPO training |

### How It Works

#### 1. Environment Detection

The `mock_utils.py` module provides utilities for detecting when to use mock scripts:

```python
from arbor.server.utils.mock_utils import should_use_mock_gpu

if should_use_mock_gpu():
    print("Using GPU mocks")
```

#### 2. Script Path Resolution

Job classes automatically select the appropriate script:

```python
from arbor.server.utils.mock_utils import get_script_path

# Returns mock version if GPU mocking is enabled
script_path = get_script_path("grpo_training.py", script_dir)
```

#### 3. Environment Setup

Mock environment variables are automatically set:

```python
from arbor.server.utils.mock_utils import setup_mock_environment

env = setup_mock_environment(os.environ.copy())
# env now contains ARBOR_MOCK_GPU=1, cleared CUDA_VISIBLE_DEVICES, etc.
```

### Testing the Mock System

Run the comprehensive test suite:

```bash
python test_gpu_mocking.py
```

This tests:
- Mock detection logic
- Script path selection
- Environment variable setup
- Mock vLLM server startup
- Mock training script execution

### Writing New Mock Scripts

When adding new GPU-intensive scripts, follow this pattern:

#### 1. Create the Mock Script

```python
# my_gpu_script_mock.py
import argparse
import time

def main():
    parser = argparse.ArgumentParser()
    # Add same arguments as real script
    parser.add_argument("--model", type=str)
    # ... other args

    args = parser.parse_args()

    print("Mock: Starting my GPU operation...")
    # Simulate work without GPU dependencies
    for i in range(3):
        print(f"Mock: Step {i+1}/3")
        time.sleep(0.1)
    print("Mock: Operation completed!")

if __name__ == "__main__":
    main()
```

#### 2. Update Job Classes

Modify the job class to use `get_script_path()`:

```python
from arbor.server.utils.mock_utils import get_script_path, setup_mock_environment

# In your job class
script_path = get_script_path("my_gpu_script.py", script_dir)
my_env = setup_mock_environment(os.environ.copy())
```

#### 3. Test the Mock

Add tests to verify your mock works correctly.

### Mock Implementation Details

#### Mock vLLM Server (`vllm_serve_mock.py`)

- Provides identical FastAPI endpoints as real vLLM server
- Returns mock completions for `/v1/completions` and `/v1/chat/completions`
- Simulates weight synchronization endpoints
- No actual model loading or GPU operations

#### Mock Training Scripts

- Accept identical CLI arguments as real scripts
- Simulate training steps with simple loops and sleep calls
- Mock all GPU-related classes (trainers, models, accelerators)
- Provide same status outputs and logging patterns

#### Environment Variable Handling

When GPU mocking is enabled:
- `CUDA_VISIBLE_DEVICES` is cleared (set to empty string)
- `ARBOR_MOCK_GPU=1` is set for subprocess detection
- `ARBOR_GPU_MOCK_MODE=1` is set as additional indicator

### Integration with Job System

The job classes (`InferenceJob`, `GRPOJob`, `FileTrainJob`) automatically:

1. **Detect mock mode** using `should_use_mock_gpu()`
2. **Select appropriate scripts** using `get_script_path()`
3. **Set up mock environment** using `setup_mock_environment()`
4. **Launch mock processes** with identical subprocess calls

### Benefits

- **No GPU Hardware Required**: Test training and inference without GPUs
- **Faster Testing**: Mock operations complete in seconds vs hours
- **Lightweight Dependencies**: Avoid installing PyTorch, vLLM, etc. in test environments
- **Identical Interfaces**: Mock scripts accept same arguments as real ones
- **Automatic Detection**: Works seamlessly with pytest and CI systems
- **Subprocess Isolation**: Only affects subprocess environments, not main process

### Troubleshooting

#### Mock Not Detected

Ensure environment variables are set:
```bash
export ARBOR_MOCK_GPU=1
```

Check detection logic:
```python
from arbor.server.utils.mock_utils import should_use_mock_gpu
print(should_use_mock_gpu())  # Should be True
```

#### Mock Script Not Found

Verify the mock script exists:
```bash
ls arbor/server/services/scripts/*_mock.py
ls arbor/server/services/inference/*_mock.py
```

#### Import Errors in Mock Scripts

Mock scripts should not import heavy dependencies. Check that your mock script only imports standard library and lightweight packages.

## Testing

### Test Suite Organization

Arbor has a comprehensive test suite organized into two main categories:

- **Unit Tests** (`tests/server/`): Fast tests using TestClient and mocking
- **Integration Tests** (`tests/integration/`): End-to-end tests with real server processes

### Running Tests

```bash
# Run all unit tests (recommended for development)
ARBOR_MOCK_GPU=1 uv run --only-group minimal pytest tests/server/ tests/test_gpu_mocking.py -v

# Run integration tests
ARBOR_MOCK_GPU=1 uv run --only-group minimal pytest tests/integration/ -v

# Run all tests
ARBOR_MOCK_GPU=1 uv run --only-group minimal pytest

# Run with coverage reporting
ARBOR_MOCK_GPU=1 uv run --only-group minimal pytest tests/server/ tests/test_gpu_mocking.py --cov=arbor --cov-report=html
```

### Test Coverage

The project uses `pytest-cov` for coverage reporting. Coverage reports help identify areas that need more testing:

- **Minimum Coverage**: 30% (enforced by pre-commit hooks)
- **Coverage Reports**: Available in terminal and HTML format (`htmlcov/`)
- **Excluded from Coverage**:
  - Test files
  - Examples directory
  - Non-mocked GPU scripts (since we test with mocks)
  - Heavy ML dependencies (vLLM, training scripts)

View coverage reports:
```bash
# Generate and view HTML coverage report
ARBOR_MOCK_GPU=1 uv run --only-group minimal pytest tests/server/ --cov=arbor --cov-report=html
open htmlcov/index.html
```

### Pre-commit Testing

Pre-commit hooks automatically run the full test suite before each commit:

```bash
# Install pre-commit hooks (one time setup)
uv run --only-group minimal pre-commit install

# Run pre-commit checks manually
uv run --only-group minimal pre-commit run --all-files
```

Pre-commit runs:
- Code formatting (Black, isort)
- File validation (YAML, JSON, TOML syntax)
- Debug statement detection
- Unit tests with coverage reporting (~1.5s)
- Integration tests (~9s)

### Writing Tests

When writing tests that involve GPU operations:

1. **Use pytest**: GPU mocking is automatically enabled
2. **Set environment**: Or manually set `ARBOR_MOCK_GPU=1`
3. **Test mock behavior**: Verify mock scripts are used instead of real ones
4. **Add cleanup**: Use `yield` in fixtures to ensure proper cleanup

Example:
```python
import os
import pytest
from arbor.server.utils.mock_utils import should_use_mock_gpu, get_script_path

def test_gpu_mocking():
    os.environ["ARBOR_MOCK_GPU"] = "1"
    assert should_use_mock_gpu()

    script_path = get_script_path("grpo_training.py", "/some/dir")
    assert "grpo_training_mock.py" in script_path

@pytest.fixture(scope="module")
def server(tmp_path_factory):
    """Server fixture with proper cleanup"""
    # Setup code...
    yield app

    # Cleanup to prevent test hanging
    try:
        app.state.job_manager.cleanup()
    except Exception as e:
        print(f"Error during cleanup: {e}")
```

### Test Performance

- **Unit Tests**: ~0.6 seconds (26 tests)
- **Integration Tests**: ~9 seconds (10 tests)
- **GPU Mocking Tests**: ~0.35 seconds (8 tests)
- **Total**: ~10-15 seconds for full suite

### Continuous Integration

GitHub Actions automatically run tests on all branches:

#### **Test Workflow** (`.github/workflows/test.yml`)
Runs on: `push` to `main`/`dev` branches, and all pull requests

- **Multi-Python Testing**: Tests against Python 3.11, 3.12, and 3.13
- **Unit Tests**: With coverage reporting (30% minimum)
- **Integration Tests**: End-to-end functionality testing
- **Coverage Upload**: Uploads coverage to Codecov (Python 3.13 only)
- **Performance**: ~2-3 minutes per Python version

#### **Code Quality Workflow** (`.github/workflows/code-quality.yml`)
Runs on: Pull requests touching Python files

- **Pre-commit Checks**: Runs all pre-commit hooks (formatting, tests, debug detection)
- **Code Formatting**: Black, isort validation
- **File Validation**: Basic configuration file validation
- **Comprehensive Quality**: All the same checks as local pre-commit
- **Performance**: ~30-60 seconds

#### **Status Checks**
Both workflows are required to pass before merging pull requests. This ensures:
- Code is properly formatted and follows style guidelines
- All tests pass across supported Python versions
- Test coverage is maintained above the minimum threshold
- No debug statements are accidentally committed

## Code Style

Use the included development tools:

```bash
# Format code
black .

# Sort imports
isort .

# Run pre-commit hooks
pre-commit run --all-files
```

---

For questions about development setup or GPU mocking, please open an issue on the GitHub repository.
