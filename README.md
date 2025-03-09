# Arbor 🌳

A drop-in replacement for OpenAI's fine-tuning API that lets you fine-tune and manage open-source language models locally. Train and deploy custom models with the same API you already know.

## Installation

```bash
pip install arbor-ai
```

## Quick Start

1. Start the Arbor server:

```bash
arbor serve
```

2. The server will be available at `http://localhost:8000`.

3. Upload your training data:

```python
import requests

requests.post('http://127.0.0.1:8000/api/files', files={'file': open('your_file.jsonl', 'rb')})
```

4. Submit a fine-tuning job:

```python
requests.post('http://127.0.0.1:8000/api/fine-tune', json={'model': 'HuggingFaceTB/SmolLM2-135M-Instruct', 'training_file': 'Returned file ID from Step 3'})
```

5. Monitor the job status:

```python
requests.get('http://127.0.0.1:8000/api/jobs/{Returned job ID from Step 4}')
```



## Development Setup

```bash
poetry install
```

```bash
poetry run arbor serve
```

```bash
poetry run pytest
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

If you encounter any issues or have questions, please file an issue on the [GitHub repository](https://github.com/Ziems/arbor/issues).