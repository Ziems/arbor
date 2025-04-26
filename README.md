# 🌳 Arbor

**A drop-in replacement for OpenAI's fine-tuning API that lets you fine-tune open-source language models locally.**
Train and deploy custom models using the same API you already know.

---

## 🚀 Installation

Install Arbor via pip:

```bash
git clone https://github.com/Ziems/arbor
cd arbor
uv pip install -e .
```

---

## ⚡ Quick Start

### 1️⃣ Start the Server

**CLI:**

```bash
uv run arbor serve
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
    'http://127.0.0.1:8000/v1/files',
    files={'file': open('your_file.jsonl', 'rb')}
)
```

The response will look something like:

```json
{
  "id": "ff66c017-7330-4da3-a60b-38fe6ef46cce",
  "object": "file",
  "bytes": 1227,
  "created_at": 1741705464,
  "filename": "training_data_sft.jsonl",
  "purpose": "training"
}
```
> 📍 **Note:** The `id` from the response is the file ID you need to use in the next step.

### 3️⃣ Submit a Fine-Tuning Job

```python
requests.post(
    'http://127.0.0.1:8000/v1/fine_tuning/jobs',
    json={'model': 'HuggingFaceTB/SmolLM2-135M-Instruct', 'training_file': 'your_file_id'}
)
```

The response will look something like:

```json
{
  "id":"24649b59-7a45-4bb8-88b7-2bd676a144f1",
  "status":"queued",
  "details":"",
  "fine_tuned_model":null
}
```
> 📍 **Note:** The `id` from the response is the job ID you need to use in the next step.

### 4️⃣ Monitor Job Status

```python
requests.get('http://127.0.0.1:8000/v1/fine_tuning/jobs/{your_job_id}')
```

While the job is running, the response will look something like:

```json
{
  "id":"24649b59-7a45-4bb8-88b7-2bd676a144f1",
  "status":"running",
  "details":"",
  "fine_tuned_model":null
}
```

When the job is complete, the response will look something like:

```json
{
  "id": "69e524cb-c682-498d-955b-d148c0866de3",
  "status": "succeeded",
  "details": "",
  "fine_tuned_model": "/home/noah/Code/OSS/arbor/storage/models/ft:smollm2-135m-instruct:inhvr6:20250311_111457"
}
```
> 📍 **Note:** The `fine_tuned_model` field will contain the full path to the fine-tuned model.

### 5️⃣ Use the Fine-Tuned Model

You can then use the fine-tuned model in the same way you would use any other model in HuggingFace

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("/home/noah/Code/OSS/arbor/storage/models/ft:smollm2-135m-instruct:inhvr6:20250311_111457")
tokenizer = AutoTokenizer.from_pretrained("/home/noah/Code/OSS/arbor/storage/models/ft:smollm2-135m-instruct:inhvr6:20250311_111457")

```

Coming soon:

- 🔄 Support for inference
- 🔄 Support for optimized resource and job management

---

## 🛠 Development Setup

Clone the repo and set up your development environment:

```bash
uv venv
source .venv/bin/activate

uv pip install -e .
uv run arbor serve
```

Run tests:

```bash
uv run pytest
```

Pre-commit for [Black Formatter](https://github.com/psf/black)

```bash
pre-commit install
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
