<p align="center">
  <img src="https://github.com/user-attachments/assets/ed0dd782-65fa-48b5-a762-b343b183be09" alt="Description" width="400"/>
</p>

**A framework for optimizing DSPy programs with RL.**

[![PyPI Downloads](https://static.pepy.tech/badge/arbor-ai/month)](https://pepy.tech/projects/arbor-ai)

[![DSPy Discord](https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white)](https://discord.gg/ZAEGgxjPUe) [![Arbor Channel](https://img.shields.io/badge/Arbor%20Channel-Open-5865F2?logo=discord&logoColor=white)](https://discordapp.com/channels/1161519468141355160/1396547082839654430)

---

## 🚀 Installation

Install Arbor via pip:

```bash
pip install -U arbor-ai
```

Optionally, you can also install flash attention to speed up inference. <br/>
This can take 15+ minutes to install on some setups:

```bash
pip install flash-attn --no-build-isolation
```

---

## ⚡ Quick Start

### 1️⃣ Create a Config

Arbor looks for a config at `~/.arbor/config.yaml` by default.

- Auto-setup (recommended): run the server once and follow prompts. It will create `~/.arbor/config.yaml`.
  ```bash
  python -m arbor.cli serve
  ```

- Manual setup: create `~/.arbor/config.yaml` with at least GPU IDs. You may omit `storage_path` to use the default (`~/.arbor/storage` locally or `/root/.arbor/storage` in Docker).

  Example:
  ```yaml
  # ~/.arbor/config.yaml
  # Optional: use absolute path; omit to use default
  # storage_path: /home/<your-user>/.arbor/storage
  inference:
    gpu_ids: [0]
  training:
    gpu_ids: [1, 2]
  ```

### 2️⃣ Start the Server

**CLI:**

```bash
python -m arbor.cli serve
```

**Docker (GPU):**

```bash
docker run --gpus all -p 7453:7453 -v ~/.arbor:/root/.arbor arbor-ai
```

- This mounts your local `~/.arbor` (which contains `config.yaml` and `storage/`) into the container at `/root/.arbor` and exposes the default port `7453`.
- If your config uses an absolute path like `/home/<user>/.arbor/storage`, either:
  - mount that same path into the container, or
  - update `storage_path` in `~/.arbor/config.yaml` to `/root/.arbor/storage`.

**Google Colab/Jupyter:**

```python
import arbor
arbor.init()  # Auto-detects GPUs, creates config, starts server in background

from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:7453/v1", api_key="not-needed")
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ziems/arbor/blob/main/examples/colab_quickstart.ipynb)

### 3️⃣ Optimize a DSPy Program

Follow the DSPy tutorials here to see usage examples:
[DSPy RL Optimization Examples](https://dspy.ai/tutorials/rl_papillon/)

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

## 🙏 Acknowledgements

Arbor builds on the shoulders of great work. We extend our thanks to:

- **[Will Brown's Verifiers library](https://github.com/willccbb/verifiers)**
- **[Hugging Face TRL library](https://github.com/huggingface/trl)**

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@misc{ziems2025arbor,
  title={Arbor: Open Source Language Model Post Training},
  author={Ziems, Noah and Agrawal, Lakshya A and Soylu, Dilara and Lai, Liheng and Miller, Isaac and Qian, Chen and Jiang, Meng and Khattab, Omar},
  howpublished = {\url{https://github.com/Ziems/arbor}},
  year={2025}
}
```
