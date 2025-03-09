import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from arbor.server.api.routes.files import router
from arbor.server.services.file_manager import FileManager
from pathlib import Path
from arbor.server.services.dependencies import get_file_manager

@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(router, prefix="/api/files")
    return app

@pytest.fixture
def client(app):
    return TestClient(app)

def test_upload_file(client):
    # Load the test file from tests/data
    test_file_path = Path(__file__).parent.parent.parent.parent / "data" / "training_data_sft.jsonl"
    test_content = test_file_path.read_bytes()
    files = {"file": ("test.jsonl", test_content, "application/json")}

    response = client.post("/api/files", files=files)

    assert response.status_code == 200
    assert "id" in response.json()

def test_upload_file_validates_file_type(client):
    # Load the test file from tests/data
    test_file_path = Path(__file__).parent.parent.parent.parent / "data" / "training_data_sft.jsonl"
    test_content = test_file_path.read_bytes()

    # Test rejected file types
    invalid_content = b"test file content"
    invalid_files = [
        ("test.txt", invalid_content, "text/plain"),
        ("test.json", b'{"key": "value"}', "application/json")
    ]

    for filename, content, content_type in invalid_files:
        files = {"file": (filename, content, content_type)}
        response = client.post("/api/files", files=files)
        assert response.status_code == 400
        assert "Only .jsonl files are allowed" in response.json()["detail"]

    # Test accepted file type
    files = {"file": ("test.jsonl", test_content, "application/json")}
    response = client.post("/api/files", files=files)
    assert response.status_code == 200
    assert "id" in response.json()

def test_upload_file_validates_format(client):
    # Valid JSONL format (from our test file)
    test_file_path = Path(__file__).parent.parent.parent.parent / "data" / "training_data_sft.jsonl"
    valid_content = test_file_path.read_bytes()

    # Test valid format
    files = {"file": ("test.jsonl", valid_content, "application/json")}
    response = client.post("/api/files", files=files)
    assert response.status_code == 200
    assert "id" in response.json()

    # Test empty file
    files = {"file": ("test.jsonl", b"", "application/json")}
    response = client.post("/api/files", files=files)
    assert response.status_code == 400
    assert "Invalid file format" in response.json()["detail"]

    # Test plain text
    files = {"file": ("test.jsonl", b"This is not a JSONL file", "application/json")}
    response = client.post("/api/files", files=files)
    assert response.status_code == 400
    assert "Invalid file format" in response.json()["detail"]

    # Test invalid JSON structure
    files = {"file": ("test.jsonl", b'{"messages": [{"role": "user"}]}\n{"broken_json":', "application/json")}
    response = client.post("/api/files", files=files)
    assert response.status_code == 400
    assert "Invalid file format" in response.json()["detail"]

    # Test missing required fields
    files = {"file": ("test.jsonl", b'{"wrong_field": "test"}\n{"also_wrong": "test"}', "application/json")}
    response = client.post("/api/files", files=files)
    assert response.status_code == 400
    assert "Invalid file format" in response.json()["detail"]

    # Test missing messages field
    files = {"file": ("test.jsonl", b'{"other": "data"}\n{"more": "data"}', "application/json")}
    response = client.post("/api/files", files=files)
    assert response.status_code == 400
    assert "Invalid file format" in response.json()["detail"]

def test_get_file_returns_same_content(client):
    # Valid JSONL format (from our test file)
    test_file_path = Path(__file__).parent.parent.parent.parent / "data" / "training_data_sft.jsonl"
    valid_content = test_file_path.read_bytes()

    # Upload file first
    files = {"file": ("test.jsonl", valid_content, "application/json")}
    response = client.post("/api/files", files=files)
    assert response.status_code == 200
    file_id = response.json()["id"]

    # Get the file and verify contents match
    file_manager = get_file_manager()
    file_data = file_manager.get_file(file_id)

    with open(file_data["path"], "rb") as f:
        stored_content = f.read()

    assert stored_content == valid_content



