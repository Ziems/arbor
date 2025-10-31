<p align="center">
  <img src="https://github.com/user-attachments/assets/ed0dd782-65fa-48b5-a762-b343b183be09" alt="Description" width="400"/>
</p>

**A framework for optimizing DSPy programs with RL.**

[![PyPI Downloads](https://static.pepy.tech/badge/arbor-ai/month)](https://pepy.tech/projects/arbor-ai)

[![ArborRL Discord](https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white)](https://discord.gg/EuU47wKsBS)

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
import random
import dspy
from datasets import load_dataset

import arbor
from arbor import ArborGRPO, ArborProvider

# Start Arbor server (starts in background)
arbor_server_info = arbor.init()

# Load a small English→French dataset
raw_dataset = load_dataset("Helsinki-NLP/opus_books", "en-fr")
raw_data = [
    dspy.Example(english=ex["translation"]["en"], french=ex["translation"]["fr"]).with_inputs("english")
    for ex in raw_dataset["train"]
][:2000]

# Train / validation split
random.Random(43).shuffle(raw_data)
trainset = raw_data[:1000]
valset = raw_data[1000:1100]

# Define the task
translate_program = dspy.Predict("english -> french")

# Connect DSPy to Arbor
provider = ArborProvider()
lm_name = "Qwen/Qwen2.5-1.5B-Instruct"
lm = dspy.LM(
    model=f"openai/arbor:{student_lm_name}",
    provider=provider,
    api_base=arbor_server_info["base_url"],
    api_key="arbor",
    max_tokens=512,
    temperature=1.0,
)
translate_program.set_lm(lm)

# Simple reward: number of unique letters in the French output
def unique_letter_reward(example, pred, trace=None) -> float:
    letters = [ch.lower() for ch in pred.french if ch.isalpha()]
    return float(len(set(letters)))

# Optimize with Arbor's GRPO trainer
compiler = ArborGRPO(
    metric=unique_letter_reward,
    num_dspy_examples_per_grpo_step=1,
    num_rollouts_per_grpo_step=8,
)

# Run optimization
optimized_translate = compiler.compile(
    student=student_translate,
    trainset=trainset,
    valset=valset,
)

print(optimized_translate(english="hello"))
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
