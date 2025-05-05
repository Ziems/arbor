<p align="center">
  <img src="https://github.com/user-attachments/assets/ed0dd782-65fa-48b5-a762-b343b183be09" alt="Description" width="400"/>
</p>

**A framework for optimizing DSPy programs with RL.**

---

## 🚀 Installation

Install Arbor via pip:

```bash
pip install git+https://github.com/Ziems/arbor.git
```

---

## ⚡ Quick Start

### 1️⃣ Make an `arbor.yaml` File

This is all dependent on your setup. Here is an example of one:
```yaml
inference:
  gpu_ids: '0'

training:
  gpu_ids: '1, 2'
```

### 2️⃣ Start the Server

**CLI:**

```bash
python -m arbor.cli serve --arbor-config arbor.yaml
```

### 3️⃣ Optimize a DSPy Program

Follow the DSPy tutorials here to see usage examples:
[DSPy RL Optimization Examples](https://dspy.ai/tutorials/rl_papillon/)

---

## 🙏 Acknowledgements

Arbor builds on the shoulders of great work. We extend our thanks to:
- **[Will Brown's Verifiers library](https://github.com/willccbb/verifiers)**
- **[Hugging Face TRL library](https://github.com/huggingface/trl)**
