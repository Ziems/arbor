# Integration Tests

These tests require real server processes and may be slow. They test end-to-end functionality including:

- Real server startup with uvicorn
- Actual HTTP requests
- Real training processes (with GPU mocking enabled via `ARBOR_MOCK_GPU=1`)
- OpenAI client compatibility

## Running Integration Tests

```bash
# Run with GPU mocking enabled (recommended for CI/testing)
ARBOR_MOCK_GPU=1 uv run --only-group dev pytest tests/integration/ -v

# Run without mocking (requires actual GPUs and may be very slow)
uv run pytest tests/integration/ -v
```

## Test Files

- `test_training_integration.py` - Full training workflow with real server processes
- `test_openai_client_integration.py` - OpenAI client compatibility tests

For faster unit tests, see the `tests/` directory which uses TestClient and mocking.
