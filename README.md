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

If you need the latest DSPy features that haven't landed on PyPI yet, install directly from the main branch:

```bash
uv pip install -U git+https://github.com/stanfordnlp/dspy.git@main
```

Optionally, you can also install flash attention to speed up inference. <br/>
This can take 15+ minutes to install on some setups:

```bash
uv pip install flash-attn --no-build-isolation
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
    # Arbor checks to make sure these match the training config
    temperature=1.0,
    top_p=1.0,
    top_k=-1,
    repetition_penalty=1.0,
    max_tokens=2048,
)
translate_program.set_lm(lm)

# Simple reward: number of unique letters in the French output
def unique_letter_reward(example, pred, trace=None) -> float:
    letters = [ch.lower() for ch in pred.french if ch.isalpha()]
    return float(len(set(letters)))

# NOTE: Training on 4 GPUs.
train_kwargs = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 24/6, # 21 (rollouts per example) * 6 (num dspy examples per grpo step) / 6 (gpus * per device batch size)
    "temperature": 1.0,
    "top_k": -1,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
    "beta": 0.00,
    "learning_rate": 1e-6,
    "gradient_checkpointing": True,
    "fp16": True,
    "lr_scheduler_type": "constant_with_warmup",
    "loss_type": "dapo",
    "max_steps": 1000,
    "report_to": "wandb",
    "log_completions": True,
    "logging_steps": 1,
    "max_prompt_length": None,
    "max_completion_length": None,
    "scale_rewards": False,
    "max_grad_norm": 1.0,
    "lora_config": {
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "r": 8,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    },
    "num_training_gpus": 3,
    "num_inference_gpus": 1,
    "weight_decay": 0.001,
}

# Optimize with Arbor's GRPO trainer
compiler = ArborGRPO(
    metric=unique_letter_reward,
    num_dspy_examples_per_grpo_step=6,
    num_rollouts_per_grpo_step=24,
    exclude_demos=True,
    num_train_steps=1000,
    num_threads=16,
    use_train_as_val=False,
    num_steps_for_val=50,
    train_kwargs=train_kwargs,
    checkpoint="single-best",
)

# Run optimization
optimized_translate = compiler.compile(
    student=student_translate,
    trainset=trainset,
    valset=valset,
)

print(optimized_translate(english="hello"))
```
---

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

---

## Community

- Join our Discord for help, updates, and discussions: [Arbor Discord](https://discord.gg/EuU47wKsBS)
- Join the DSPy Discord for help, updates, and discussion on DSPy: [DSPy Discord](https://discord.gg/ZAEGgxjPUe)

---

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
