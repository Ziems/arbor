# Arbor Deployment Guide

This directory contains SkyPilot configurations for running Arbor GRPO training jobs on cloud infrastructure.

## Quick Start

### Prerequisites

- [SkyPilot](https://skypilot.readthedocs.io/) installed: `pip install skypilot[aws]` (or your cloud provider)
- AWS/GCP/other cloud credentials configured
- An S3 bucket (or equivalent) for storing checkpoints

### Running Any Training Script

The `skypilot_dev.yaml` configuration is designed to work with any Python training script. Scripts run from the project root directory, so use paths relative to root (e.g., `examples/your_script.py`).

#### Step 1: Configure Your Training Script

You can specify which script to run in two ways:

**Option A: Edit the YAML file (recommended)**

Open `deploy/skypilot_dev.yaml` and find line 54. Change the script path:

```yaml
# Line 54 in skypilot_dev.yaml - change the script path
TRAINING_SCRIPT=${TRAINING_SCRIPT:-examples/your_script.py}
```

**Important:** The script runs from the project root directory. Use the relative path from the root:
- Scripts in `examples/`: `examples/your_script.py`
- Scripts in root: `your_script.py`

**Examples:**
- For `examples/banking_grpo.py`: `TRAINING_SCRIPT=${TRAINING_SCRIPT:-examples/banking_grpo.py}`
- For `examples/unique_chars_grpo.py`: `TRAINING_SCRIPT=${TRAINING_SCRIPT:-examples/unique_chars_grpo.py}`
- For a script in root `train.py`: `TRAINING_SCRIPT=${TRAINING_SCRIPT:-train.py}`
- Keep default: `TRAINING_SCRIPT=${TRAINING_SCRIPT:-examples/multihop_dev.py}`

**Option B: Pass via command line (no file editing needed)**

```bash
sky jobs launch -n training-job deploy/skypilot_dev.yaml \
  --env TRAINING_SCRIPT=examples/your_script.py \
  --infra aws
```

**Example:**
```bash
sky jobs launch -n banking-training deploy/skypilot_dev.yaml \
  --env TRAINING_SCRIPT=examples/banking_grpo.py \
  --infra aws
```

#### Step 2: Configure Your Checkpoint S3 Bucket

Edit `skypilot_dev.yaml` and update the checkpoint bucket path (around line 25):

```yaml
file_mounts:
  /mnt/checkpoints:
    source: s3://your-bucket-name/  # Change this to your S3 bucket
    mode: MOUNT_CACHED
```

Replace `s3://your-bucket-name/` with your actual S3 bucket path. The bucket should already exist and be accessible with your cloud credentials.

#### Step 3: Customize Setup (if needed)

If your script requires additional dependencies or data downloads, edit the `setup` section in `skypilot_dev.yaml`:

```yaml
setup: |
  # ... common setup ...
  
  # Add your script-specific dependencies:
  uv pip install package-name
  
  # Add your script-specific data downloads:
  curl -Ls https://example.com/data.tar.gz | tar -xz -C examples
```

**Example for `multihop_dev.py`:**
```yaml
setup: |
  # ... common setup ...
  uv pip install ujson bm25s PyStemmer "jax[cpu]"
  curl -Ls https://huggingface.co/dspy/cache/resolve/main/wiki.abstracts.2017.tar.gz | tar -xz -C examples
```

#### Step 4: Launch the Training Job

```bash
sky jobs launch -n training-job deploy/skypilot_dev.yaml --infra <aws|gcp|nebius|lambda>
```

Replace `<aws|gcp|nebius|lambda>` with your cloud provider.

#### Step 5: Monitor Progress

```bash
# View logs
sky logs training-job

# Check job status
sky status
```

#### Step 6: Stop the Job (if needed)

```bash
sky jobs cancel training-job
```

## Configuration Details

### Resources

- **GPUs**: 4x L40S GPUs by default (adjust based on your script's needs)
- **Spot Instances**: Disabled by default (`use_spot: false`). Set to `true` for cost savings
- **Disk**: 256 GB

### DeepSpeed Configuration

The training uses DeepSpeed ZeRO-3 (configured in `../configs/deepspeed_config.yaml`):
- ZeRO Stage 3 for efficient memory management
- Mixed precision: bf16
- Optimized for multi-node distributed training (currently single-node)

The script will automatically use accelerate launch with DeepSpeed if:
- You set `USE_ACCELERATE=true` environment variable, OR
- The script name is `multihop_dev.py` (default)

Otherwise, it runs the script directly with `python`.

### What Gets Installed

The setup script automatically installs:
- Project dependencies via `uv sync`
- Common packages: `datasets`, `deepspeed`, `accelerate`

You can add script-specific dependencies in the setup section.

## Examples

### Running multihop_dev.py

1. Edit `skypilot_dev.yaml` setup section to uncomment multihop-specific setup:
```yaml
uv pip install ujson bm25s PyStemmer "jax[cpu]"
curl -Ls https://huggingface.co/dspy/cache/resolve/main/wiki.abstracts.2017.tar.gz | tar -xz -C examples
```

2. The script path is already set to `examples/multihop_dev.py` by default

3. Launch:
```bash
sky jobs launch -n multihop-dev deploy/skypilot_dev.yaml --infra aws
```

### Running banking_grpo.py

1. Edit the script path in the run section:
```yaml
TRAINING_SCRIPT=${TRAINING_SCRIPT:-examples/banking_grpo.py}
```

2. Launch (without accelerate, since it doesn't need distributed training):
```bash
sky jobs launch -n banking-training deploy/skypilot_dev.yaml --infra aws
```

Or if you want to use accelerate anyway:
```bash
sky jobs launch -n banking-training deploy/skypilot_dev.yaml \
  --env TRAINING_SCRIPT=examples/banking_grpo.py \
  --env USE_ACCELERATE=true \
  --infra aws
```

No additional setup needed - it only uses standard dependencies.

### Running unique_chars_grpo.py

Same as banking_grpo.py - just change the script name.

## Customization

### Changing GPU Count

To use a different number of GPUs, modify `skypilot_dev.yaml`:

```yaml
resources:
  accelerators: L40S:4  # Change number as needed
```

**Note**: If you change the GPU count and your script uses accelerate, make sure the number matches your script's expectations.

### Using Spot Instances

For cost savings, enable spot instances:

```yaml
resources:
  use_spot: true
```

### Forcing Accelerate Launch

If you want to force accelerate launch even if your script doesn't explicitly use it:

```bash
sky jobs launch -n training-job deploy/skypilot_dev.yaml \
  --env USE_ACCELERATE=true \
  --infra aws
```

### Changing Cloud Providers

SkyPilot supports multiple providers. Specify during launch:

```bash
sky jobs launch -n training-job deploy/skypilot_dev.yaml --infra aws
sky jobs launch -n training-job deploy/skypilot_dev.yaml --infra gcp
sky jobs launch -n training-job deploy/skypilot_dev.yaml --infra azure
```

## Troubleshooting

### Checkpoint Access Issues

If you see errors accessing the S3 bucket:
1. Verify your AWS credentials are configured: `aws s3 ls s3://your-bucket-name/`
2. Ensure the bucket exists and is accessible
3. Check that SkyPilot has the necessary IAM permissions

### Script Not Found

If you get "script not found" errors:
1. The script runs from the project root directory - verify your path is correct relative to root
2. Scripts in `examples/` should use: `examples/your_script.py`
3. Scripts in root should use: `your_script.py`
4. Check that the path matches exactly (including `.py` extension)
5. Ensure the path doesn't have typos

### Missing Dependencies

If your script fails due to missing packages:
1. Add the required packages to the `setup` section in `skypilot_dev.yaml`
2. Re-launch the job

### NCCL Errors

If you encounter NCCL communication errors, add these environment variables to the `run` section:

```yaml
run: |
  export NCCL_P2P_DISABLE=1
  export NCCL_IB_DISABLE=1
  # ... rest of run script
```

### GPU Memory Issues

If you run out of GPU memory:
- Reduce batch sizes in your training script
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Ensure DeepSpeed ZeRO-3 is properly configured (if using accelerate)

## Additional Resources

- [SkyPilot Documentation](https://skypilot.readthedocs.io/)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [Arbor Documentation](../README.md)
