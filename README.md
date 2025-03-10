# 🌳 Arbor 

**A drop-in replacement for OpenAI's fine-tuning API that lets you fine-tune open-source language models locally.**  
Train and deploy custom models using the same API you already know.

---

## 🚀 Installation

Install Arbor via pip:

```bash
pip install arbor-ai
```

---

## ⚡ Quick Start

### 1️⃣ Start the Server

**CLI:**

```bash
arbor serve
```

**Python:**

```python
from arbor import start_server, stop_server

server = start_server()
# Your fine-tuning operations here...
stop_server(server)
```

> 📍 **By default, the server runs at** `http://localhost:8000`

### 2️⃣ Upload Training Data

```python
import requests

requests.post(
    'http://127.0.0.1:8000/api/files', 
    files={'file': open('your_file.jsonl', 'rb')}
)
```

### 3️⃣ Submit a Fine-Tuning Job

```python
requests.post(
    'http://127.0.0.1:8000/api/fine-tune',
    json={'model': 'HuggingFaceTB/SmolLM2-135M-Instruct', 'training_file': 'your_file_id'}
)
```

### 4️⃣ Monitor Job Status

```python
requests.get('http://127.0.0.1:8000/api/jobs/{your_job_id}')
```

---

## 🛠 Development Setup

Clone the repo and set up your development environment:

```bash
poetry install
poetry run arbor serve
```

Run tests:

```bash
poetry run pytest
```

---

## 🤝 Contributing

We welcome contributions!  
Feel free to submit a Pull Request or open an issue. 🚀

---

## 📜 License

Licensed under the **MIT License** – see the [LICENSE](./LICENSE) file for details.

---

## ❓ Support

If you encounter any issues or have questions, please file an issue on the [GitHub repository](https://github.com/Ziems/arbor/issues).  
We’re happy to help! 💡
