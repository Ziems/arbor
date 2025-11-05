# Arbor Deployment Guide

This directory contains SkyPilot configurations for running Arbor GRPO training jobs on cloud infrastructure.

## Quick Start

### Prerequisites

- [SkyPilot](https://skypilot.readthedocs.io/) installed: `pip install skypilot[aws]` (or your cloud provider)
- AWS/GCP/other cloud credentials configured
- An S3 bucket (or equivalent) for storing checkpoints

### Running Any Training Script

The `skypilot_example.yaml` configuration is designed to work with any Python training script. Scripts run from the project root directory, so use paths relative to root (e.g., `examples/your_script.py`).

#### Step 1: Configure Your Training Script

```bash
sky jobs launch -n training-job deploy/skypilot_example.yaml \
  --env TRAINING_SCRIPT=examples/your_script.py \
  --infra aws
```

**Example:**
```bash
sky jobs launch -n banking-training deploy/skypilot_example.yaml \
  --env TRAINING_SCRIPT=examples/banking_grpo.py \
  --infra aws
```

#### Step 2: Configure Your Checkpoint S3 Bucket

Edit `skypilot_example.yaml` and update the checkpoint bucket path (around line 25):

```yaml
file_mounts:
  /mnt/checkpoints:
    source: s3://your-bucket-name/  # Change this to your S3 bucket
    mode: MOUNT_CACHED
```

Replace `s3://your-bucket-name/` with your actual S3 bucket path. The bucket should already exist and be accessible with your cloud credentials.

#### Step 3: Customize Setup (if needed)

If your script requires additional dependencies or data downloads, edit the `setup` section in `skypilot_example.yaml`:

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
sky jobs launch -n training-job deploy/skypilot_example.yaml --infra <aws|gcp|nebius|lambda>
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

### What Gets Installed

The setup script automatically installs:
- Project dependencies via `uv sync`
- Common packages: `datasets`, `deepspeed`, `accelerate`

You can add script-specific dependencies in the setup section.

## Examples

### Running multihop_dev.py

1. Edit `skypilot_example.yaml` setup section to uncomment multihop-specific setup:
```yaml
uv pip install ujson bm25s PyStemmer "jax[cpu]"
curl -Ls https://huggingface.co/dspy/cache/resolve/main/wiki.abstracts.2017.tar.gz | tar -xz -C examples
```

2. The script path is already set to `examples/multihop_dev.py` by default

3. Launch:
```bash
sky jobs launch -n multihop-dev deploy/skypilot_example.yaml --infra aws
```

### Running banking_grpo.py

1. Edit the script path in the run section:
```yaml
TRAINING_SCRIPT=${TRAINING_SCRIPT:-examples/banking_grpo.py}
```

2. Launch (without accelerate, since it doesn't need distributed training):
```bash
sky jobs launch -n banking-training deploy/skypilot_example.yaml --infra aws
```

Or if you want to use accelerate anyway:
```bash
sky jobs launch -n banking-training deploy/skypilot_example.yaml \
  --env TRAINING_SCRIPT=examples/banking_grpo.py \
  --env USE_ACCELERATE=true \
  --infra aws
```

No additional setup needed - it only uses standard dependencies.

## Customization

### Changing GPU Count

To use a different number of GPUs, modify `skypilot_example.yaml`:

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

### Changing Cloud Providers

SkyPilot supports multiple providers. Specify during launch:

```bash
sky jobs launch -n training-job deploy/skypilot_example.yaml --infra aws
sky jobs launch -n training-job deploy/skypilot_example.yaml --infra gcp
sky jobs launch -n training-job deploy/skypilot_example.yaml --infra azure
```



## Additional Resources

- [SkyPilot Documentation](https://skypilot.readthedocs.io/)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [Arbor Documentation](../README.md)
