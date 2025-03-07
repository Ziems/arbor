# Arbor AI

## Setup

```bash
poetry install
```

```bash
poetry run arbor serve
```

## Uploading Data

```bash
curl -X POST "http://localhost:8000/api/files" -F "file=@training_data.jsonl"
```
