<p align="center">
  <img src="https://github.com/user-attachments/assets/ed0dd782-65fa-48b5-a762-b343b183be09" alt="Description" width="400"/>
</p>

**A framework for optimizing DSPy programs with RL.**

[![PyPI Downloads](https://static.pepy.tech/badge/arbor-ai/month)](https://pepy.tech/projects/arbor-ai)

[![DSPy Discord](https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white)](https://discord.gg/ZAEGgxjPUe) [![Arbor Channel](https://img.shields.io/badge/Arbor%20Channel-Open-5865F2?logo=discord&logoColor=white)](https://discordapp.com/channels/1161519468141355160/1396547082839654430)

---

## 🚀 Installation

Install Arbor via uv (recommended) or pip:

```bash
uv pip install -U arbor-ai
# or: pip install -U arbor-ai
```

Optionally, you can also install flash attention to speed up inference. <br/>
This can take 15+ minutes to install on some setups:

```bash
uv pip install flash-attn --no-build-isolation
# or: pip install flash-attn --no-build-isolation
```

---

## ⚡ Quick Start

### Optimize a DSPy Program with RL

```python
import arbor
import dspy
import random

# Start Arbor server (auto-detects GPUs, starts in background)
arbor.init()

# Sample classification data
data = [
    dspy.Example(text="I want to transfer money", label="transfer").with_inputs("text"),
    dspy.Example(text="What is my balance?", label="balance").with_inputs("text"),
    dspy.Example(text="I lost my credit card", label="card_issues").with_inputs("text"),
    # ... more examples
]

# Split into train/validation
random.Random(42).shuffle(data)
trainset, valset = data[:6], data[6:]

# Define classification task
CLASSES = ["transfer", "balance", "card_issues", "pin_change"]
classify = dspy.ChainOfThought(f"text -> label: Literal{CLASSES}")

# Set up DSPy with Arbor backend
from dspy.clients.lm_local_arbor import ArborProvider
provider = ArborProvider()
student_lm = dspy.LM(
    model="openai/arbor:Qwen/Qwen2-0.5B-Instruct",
    provider=provider,
    api_base="http://127.0.0.1:7453/v1/",
    api_key="arbor"
)

student_classify = classify.deepcopy()
student_classify.set_lm(student_lm)

# Optimize with GRPO (requires 2+ GPUs)
from dspy.teleprompt.grpo import GRPO

compiler = GRPO(
    metric=lambda x, y: x.label == y.label,
)

# Run optimization
optimized_classify = compiler.compile(
    student=student_classify,
    trainset=trainset,
    valset=valset
)

# Your classifier is now optimized with RL! 🎉
```


### Traditional Fine-tuning (SFT & DPO)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ziems/arbor/blob/main/examples/colab_quickstart.ipynb)

Arbor also supports standard fine-tuning with OpenAI-compatible API:

```python
import arbor
from openai import OpenAI

# Start Arbor
arbor.init()
client = arbor.get_client()

# Upload training data
with open("sft_data.jsonl", "rb") as f:
    file = client.files.create(file=f, purpose="fine-tune")

# SFT (Supervised Fine-Tuning)
job = client.fine_tuning.jobs.create(
    model="Qwen/Qwen2-0.5B-Instruct",
    training_file=file.id,
    method={"type": "sft"}
)

# DPO (Direct Preference Optimization)
job = client.fine_tuning.jobs.create(
    model="Qwen/Qwen2-0.5B-Instruct",
    training_file=file.id,
    method={"type": "dpo"}
)

# Monitor training
arbor.watch_job(job.id)
```

### Remote GPU Setup (Alternative)

For remote servers or custom configurations, use the CLI approach:

**1. Create config at `~/.arbor/config.yaml`:**
```yaml
storage_path: ~/.arbor/storage
```

The server will automatically detect available GPUs at startup.

**2. Start server:**
```bash
uv run arbor serve
```

**3. Connect from remote:**
```python
from openai import OpenAI
client = OpenAI(base_url="http://your-server:7453/v1", api_key="not-needed")
```

**Docker deployment:**
```bash
docker run --gpus all -p 7453:7453 -v ~/.arbor:/root/.arbor arbor-ai
```

---

### Accelerate Configuration

For advanced distributed training setups, you can specify a custom [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/) config in your `~/.arbor/config.yaml`:

```yaml
accelerate_config: "/path/to/your/accelerate_config.yaml"
```

---

### Troubleshooting

**NCCL Errors**
Certain GPU setups, particularly with newer GPUs, seem to have issues with NCCL that cause Arbor to crash. Often times of these can be fixed with the following environment variables:

```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
```

**NVCC**
If you run into issues, double check that you have [nvcc](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/) installed:

```bash
nvcc --version
```

If you don't have admin permissions, you can often install nvcc using conda.

---

## Community

- Join our Discord for help, updates, and discussion: [DSPy Discord](https://discord.gg/ZAEGgxjPUe)
- Arbor-specific channel in the DSPy Discord: [Arbor Channel](https://discordapp.com/channels/1161519468141355160/1396547082839654430)

---

## 🛠️ Development workflow

To avoid merge surprises, make sure the `dev` branch is rebased on the
latest `main` history before integrating feature work. A quick sanity
check is to fetch the remote `main` ref and merge it locally:

```bash
git fetch origin main
git checkout dev
git merge origin/main
```

Only proceed with your feature merges once the commands above succeed.

## 🙏 Acknowledgements

Arbor builds on the shoulders of great work. We extend our thanks to:

- **[Will Brown's Verifiers library](https://github.com/willccbb/verifiers)**
- **[Hugging Face TRL library](https://github.com/huggingface/trl)**

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{ziems2025multi,
  title={Multi-module GRPO: Composing policy gradients and prompt optimization for language model programs},
  author={Ziems, Noah and Soylu, Dilara and Agrawal, Lakshya A and Miller, Isaac and Lai, Liheng and Qian, Chen and Song, Kaiqiang and Jiang, Meng and Klein, Dan and Zaharia, Matei and others},
  journal={arXiv preprint arXiv:2508.04660},
  year={2025}
}
```

```bibtex
@article{agrawal2025gepa,
  title={GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning},
  author={Agrawal, Lakshya A and Tan, Shangyin and Soylu, Dilara and Ziems, Noah and Khare, Rishi and Opsahl-Ong, Krista and Singhvi, Arnav and Shandilya, Herumb and Ryan, Michael J and Jiang, Meng and others},
  journal={arXiv preprint arXiv:2507.19457},
  year={2025}
}
```

```bibtex
@misc{ziems2025arbor,
  title={Arbor: Open Source Language Model Post Training},
  author={Ziems, Noah and Agrawal, Lakshya A and Soylu, Dilara and Lai, Liheng and Miller, Isaac and Qian, Chen and Jiang, Meng and Khattab, Omar},
  howpublished = {\url{https://github.com/Ziems/arbor}},
  year={2025}
}
```
