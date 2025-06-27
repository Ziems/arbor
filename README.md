<p align="center">
  <img src="https://github.com/user-attachments/assets/ed0dd782-65fa-48b5-a762-b343b183be09" alt="Description" width="400"/>
</p>

**A framework for optimizing DSPy programs with RL.**

[![PyPI Downloads](https://static.pepy.tech/badge/arbor-ai/month)](https://pepy.tech/projects/arbor-ai)

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

### 1️⃣ Start the Server

**CLI:**

```bash
python -m arbor.cli serve
```
On the first run you'll be asked which GPUs will be used for training and which for inference. For more that one GPU, separate the ids by comma: `1, 2`. Your config file will be saved in `~/.arbor/config.yaml` should you want to edit these configs in the future.

### 2️⃣ Optimize a DSPy Program

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
  year={2025}
}
```
