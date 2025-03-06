# DSPy Trainer

## Setup

```bash
poetry install
```

```bash
poetry run dspy-trainer serve
```

## Uploading Data

```bash
curl -X POST "http://localhost:8000/api/files" -F "file=@training_data.jsonl"
```
