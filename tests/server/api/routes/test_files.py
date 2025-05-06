from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from arbor.server.api.routes.files import router
from arbor.server.services.file_manager import FileManager


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    """Set up a test server with configured dependencies"""
    from arbor.server.core.config import Settings
    from arbor.server.main import app
    from arbor.server.services.file_manager import FileManager
    from arbor.server.services.job_manager import JobManager
    from arbor.server.services.training_manager import TrainingManager

    # Use tmp_path_factory instead of tmp_path because we're using scope="module"
    test_storage = tmp_path_factory.mktemp("test_storage")

    # Create test settings
    settings = Settings(STORAGE_PATH=str(test_storage))

    # Initialize services with test settings
    file_manager = FileManager(settings=settings)
    job_manager = JobManager(settings=settings)
    training_manager = TrainingManager(settings=settings)

    # Inject dependencies into app state
    app.state.settings = settings
    app.state.file_manager = file_manager
    app.state.job_manager = job_manager
    app.state.training_manager = training_manager

    return app


@pytest.fixture
def client(server):
    return TestClient(server)


def test_upload_file(client):
    # Load the test file from tests/data
    test_file_path = (
        Path(__file__).parent.parent.parent.parent / "data" / "training_data_sft.jsonl"
    )
    test_content = test_file_path.read_bytes()
    files = {"file": ("test.jsonl", test_content, "application/json")}

    response = client.post("/v1/files", files=files)

    assert response.status_code == 200
    assert "id" in response.json()


def test_upload_file_validates_file_type(client):
    # Load the test file from tests/data
    test_file_path = (
        Path(__file__).parent.parent.parent.parent / "data" / "training_data_sft.jsonl"
    )
    test_content = test_file_path.read_bytes()

    # Test rejected file types
    invalid_content = b"test file content"
    invalid_files = [
        ("test.txt", invalid_content, "text/plain"),
        ("test.json", b'{"key": "value"}', "application/json"),
    ]

    for filename, content, content_type in invalid_files:
        files = {"file": (filename, content, content_type)}
        response = client.post("/v1/files", files=files)
        assert response.status_code == 400
        assert "Only .jsonl files are allowed" in response.json()["detail"]

    # Test accepted file type
    files = {"file": ("test.jsonl", test_content, "application/json")}
    response = client.post("/v1/files", files=files)
    assert response.status_code == 200
    assert "id" in response.json()


def test_upload_file_validates_format(client):
    # Valid JSONL format (from our test file)
    test_file_path = (
        Path(__file__).parent.parent.parent.parent / "data" / "training_data_sft.jsonl"
    )
    valid_content = test_file_path.read_bytes()

    # Test valid format
    files = {"file": ("test.jsonl", valid_content, "application/json")}
    response = client.post("/v1/files", files=files)
    assert response.status_code == 200
    assert "id" in response.json()

    # Test empty file
    files = {"file": ("test.jsonl", b"", "application/json")}
    response = client.post("/v1/files", files=files)
    assert response.status_code == 400
    assert "Invalid file format" in response.json()["detail"]

    # Test plain text
    files = {"file": ("test.jsonl", b"This is not a JSONL file", "application/json")}
    response = client.post("/v1/files", files=files)
    assert response.status_code == 400
    assert "Invalid file format" in response.json()["detail"]

    # Test invalid JSON structure
    files = {
        "file": (
            "test.jsonl",
            b'{"messages": [{"role": "user"}]}\n{"broken_json":',
            "application/json",
        )
    }
    response = client.post("/v1/files", files=files)
    assert response.status_code == 400
    assert "Invalid file format" in response.json()["detail"]

    # Test missing required fields
    files = {
        "file": (
            "test.jsonl",
            b'{"wrong_field": "test"}\n{"also_wrong": "test"}',
            "application/json",
        )
    }
    response = client.post("/v1/files", files=files)
    assert response.status_code == 400
    assert "Invalid file format" in response.json()["detail"]

    # Test missing messages field
    files = {
        "file": (
            "test.jsonl",
            b'{"other": "data"}\n{"more": "data"}',
            "application/json",
        )
    }
    response = client.post("/v1/files", files=files)
    assert response.status_code == 400
    assert "Invalid file format" in response.json()["detail"]


def test_get_file_returns_same_content(client):
    # Valid JSONL format (from our test file)
    test_file_path = (
        Path(__file__).parent.parent.parent.parent / "data" / "training_data_sft.jsonl"
    )
    valid_content = test_file_path.read_bytes()

    # Upload file first
    files = {"file": ("test.jsonl", valid_content, "application/json")}
    response = client.post("/v1/files", files=files)
    assert response.status_code == 200
    file_id = response.json()["id"]

    # Get the file and verify contents match
    file_manager = client.app.state.file_manager
    file_data = file_manager.get_file(file_id)

    with open(file_data["path"], "rb") as f:
        stored_content = f.read()

    assert stored_content == valid_content
