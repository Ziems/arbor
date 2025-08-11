# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Testing
```bash
# Run all unit tests with GPU mocking (fastest for development)
ARBOR_MOCK_GPU=1 uv run --only-group minimal pytest tests/server/ tests/test_gpu_mocking.py -v

# Run integration tests
ARBOR_MOCK_GPU=1 uv run --only-group minimal pytest tests/integration/ -v

# Run all tests
ARBOR_MOCK_GPU=1 uv run --only-group minimal pytest

# Run with coverage reporting
ARBOR_MOCK_GPU=1 uv run --only-group minimal pytest tests/server/ tests/test_gpu_mocking.py --cov=arbor --cov-report=html

# Single test file
ARBOR_MOCK_GPU=1 uv run --only-group minimal pytest tests/server/test_specific.py -v
```

### Code Quality
```bash
# Format code
black .

# Sort imports
isort .

# Run all pre-commit hooks
pre-commit run --all-files
```

### Development Environment
```bash
# Minimal development environment (no heavy ML dependencies)
uv sync --only-group minimal
source .venv/bin/activate

# Full environment with GPU dependencies
uv sync --dev
source .venv/bin/activate
```

### Server Operations
```bash
# Start the Arbor server
python -m arbor.cli serve

# Start server with specific config
python -m arbor.cli serve --arbor-config ~/.arbor/config.yaml

# Docker deployment
docker run --gpus all -p 7453:7453 -v ~/.arbor:/root/.arbor arbor-ai
```

### Colab/Jupyter Integration
```python
# Ray-like interface for notebooks
import arbor

# Start server in background (auto-detects GPUs, creates config)
server_info = arbor.init()

# Get pre-configured OpenAI client
client = arbor.get_client()

# Start training job
job = client.fine_tuning.jobs.create(...)

# Watch job with real-time logs (new!)
arbor.watch_job(job.id)

# Check server status
status = arbor.status()

# Shutdown when done (optional - auto-cleanup on exit)
arbor.shutdown()
```

## Architecture Overview

### Core Components

**FastAPI Server** (`arbor/server/main.py`)
- REST API with lifespan management for resource cleanup
- Routes for files, jobs, inference, GRPO training, and monitoring

**Manager Pattern**
- All major functionality is organized into manager classes
- Base manager (`base_manager.py`) provides common functionality
- Key managers: `GPUManager`, `JobManager`, `InferenceManager`, `GRPOManager`, `FileManager`, `FileTrainManager`

**Configuration System** (`arbor/server/core/config.py`)
- YAML-based configuration loaded from `~/.arbor/config.yaml`
- Default storage path: `~/.arbor/storage` (local) or `/root/.arbor/storage` (Docker)
- GPU allocation configuration for training and inference

### Job System Architecture

**Job Lifecycle**
1. Jobs are created through API endpoints
2. `JobManager` coordinates execution across managers
3. GPU resources are allocated via `GPUManager`
4. Training/inference scripts run as subprocesses
5. Job status and logs are tracked throughout execution

**Training Types**
- **SFT (Supervised Fine-Tuning)**: Standard fine-tuning on instruction datasets
- **GRPO**: Reinforcement learning from preferences
- **DPO**: Direct Preference Optimization
- **Multi-modal GRPO**: GRPO with vision capabilities

### GPU Mocking System

**Purpose**: Test GPU-intensive operations without actual hardware or heavy ML dependencies

**Key Features**:
- Automatically enabled in test environments (`pytest`, CI)
- Mock scripts replace real GPU training/inference scripts
- Identical CLI interfaces between real and mock versions
- Controlled via `ARBOR_MOCK_GPU=1` environment variable

**Mock Scripts Location**: `arbor/server/services/scripts/*_mock.py` and `arbor/server/services/inference/*_mock.py`

### API Structure

**Endpoints**:
- `/v1/files/` - File upload and management
- `/v1/fine_tuning/jobs/` - Training job management
- `/v1/fine_tuning/grpo/` - GRPO-specific operations
- `/v1/chat/` - Inference endpoints
- `/` - Health monitoring and system status

## Important Development Notes

### Testing Requirements
- Always set `ARBOR_MOCK_GPU=1` when running tests to avoid GPU dependencies
- Use `--only-group minimal` to avoid installing heavy ML libraries during development
- Coverage minimum: 30% (enforced by pre-commit hooks)

### Code Standards
- Pre-commit hooks run automatically and include formatting, linting, and tests
- Tests must pass before commits are allowed
- Use GPU mocking for any tests involving training or inference

### Environment Variables
- `ARBOR_MOCK_GPU=1`: Enable GPU mocking system
- `CUDA_VISIBLE_DEVICES`: Cleared in mock environments
- `PYTEST_CURRENT_TEST`: Auto-detected for test mode

### Configuration
- Default config location: `~/.arbor/config.yaml`
- Required fields: `gpu_ids` for training and inference
- Optional fields: `storage_path`, `accelerate_config`, `inactivity_timeout`
- Colab auto-config: `arbor.init()` creates config automatically

### Colab Features
- **Ray-like interface**: `arbor.init()`, `arbor.shutdown()`, `arbor.status()`
- **Auto-detection**: GPUs, environment (Colab vs Jupyter), available ports
- **Background execution**: Server runs in daemon thread, survives cell execution
- **Auto-cleanup**: Server shuts down when notebook session ends
- **Pre-configured clients**: `arbor.get_client()` returns ready-to-use OpenAI client
- **Real-time log monitoring**: `arbor.watch_job()` streams training logs live
- **Colab-specific defaults**: Storage in `/content/.arbor`, security-focused host binding
