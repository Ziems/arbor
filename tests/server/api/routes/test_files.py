from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    """Set up a test server with configured dependencies"""
    from arbor.server.core.config import Config
    from arbor.server.main import app
    from arbor.server.services.managers.file_manager import FileManager
    from arbor.server.services.managers.file_train_manager import FileTrainManager
    from arbor.server.services.managers.job_manager import JobManager

    # Use tmp_path_factory instead of tmp_path because we're using scope="module"
    test_storage = tmp_path_factory.mktemp("test_storage")

    # Create test config
    from arbor.server.core.config import ArborConfig, InferenceConfig, TrainingConfig

    config = Config(
        STORAGE_PATH=str(test_storage),
        arbor_config=ArborConfig(
            inference=InferenceConfig(gpu_ids=[]), training=TrainingConfig(gpu_ids=[])
        ),
    )

    # Initialize services with test config
    file_manager = FileManager(config=config)
    job_manager = JobManager(config=config)
    file_train_manager = FileTrainManager(config=config)

    # Inject dependencies into app state
    app.state.config = config
    app.state.file_manager = file_manager
    app.state.job_manager = job_manager
    app.state.file_train_manager = file_train_manager

    yield app

    # Cleanup after all tests in this module
    try:
        app.state.job_manager.cleanup()
    except Exception as e:
        print(f"Error during job manager cleanup: {e}")


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


def test_upload_file_validates_sft_format(client):
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


def test_upload_file_validates_dpo_format(client):
    """Test DPO format validation"""
    # Valid DPO format (from our test file)
    test_file_path = (
        Path(__file__).parent.parent.parent.parent / "data" / "training_data_dpo.jsonl"
    )
    valid_content = test_file_path.read_bytes()

    # Test valid DPO format
    files = {"file": ("test.jsonl", valid_content, "application/json")}
    response = client.post("/v1/files", files=files)
    assert response.status_code == 200
    assert "id" in response.json()

    # Test DPO format with missing required fields
    files = {
        "file": (
            "test.jsonl",
            b'{"chosen": [{"role": "user", "content": "test"}], "rejected": [{"role": "assistant", "content": "test"}]}\n',
            "application/json",
        )
    }
    response = client.post("/v1/files", files=files)
    assert response.status_code == 400
    assert "Invalid file format" in response.json()["detail"]

    # Test DPO format with missing input field
    files = {
        "file": (
            "test.jsonl",
            b'{"preferred_output": [{"role": "assistant", "content": "test"}], "non_preferred_output": [{"role": "assistant", "content": "test"}]}\n',
            "application/json",
        )
    }
    response = client.post("/v1/files", files=files)
    assert response.status_code == 400
    assert "Invalid file format" in response.json()["detail"]

    # Test DPO format with invalid input structure
    files = {
        "file": (
            "test.jsonl",
            b'{"input": "not_a_dict", "preferred_output": [{"role": "assistant", "content": "test"}], "non_preferred_output": [{"role": "assistant", "content": "test"}]}\n',
            "application/json",
        )
    }
    response = client.post("/v1/files", files=files)
    assert response.status_code == 400
    assert "Invalid file format" in response.json()["detail"]

    # Test DPO format with missing tools field
    files = {
        "file": (
            "test.jsonl",
            b'{"input": {"messages": [{"role": "user", "content": "test"}]}, "preferred_output": [{"role": "assistant", "content": "test"}], "non_preferred_output": [{"role": "assistant", "content": "test"}]}\n',
            "application/json",
        )
    }
    response = client.post("/v1/files", files=files)
    assert response.status_code == 400
    assert "Invalid file format" in response.json()["detail"]

    # Test DPO format with missing parallel_tool_calls field
    files = {
        "file": (
            "test.jsonl",
            b'{"input": {"messages": [{"role": "user", "content": "test"}], "tools": []}, "preferred_output": [{"role": "assistant", "content": "test"}], "non_preferred_output": [{"role": "assistant", "content": "test"}]}\n',
            "application/json",
        )
    }
    response = client.post("/v1/files", files=files)
    assert response.status_code == 400
    assert "Invalid file format" in response.json()["detail"]


def test_upload_file_validates_mixed_formats(client):
    """Test that files with mixed or unknown formats are rejected"""
    # Test file with neither SFT nor DPO format
    files = {
        "file": (
            "test.jsonl",
            b'{"random_field": "value"}\n{"another_field": "value"}',
            "application/json",
        )
    }
    response = client.post("/v1/files", files=files)
    assert response.status_code == 400
    assert "Invalid file format" in response.json()["detail"]

    # Test file with empty JSON objects
    files = {
        "file": (
            "test.jsonl",
            b"{}\n{}",
            "application/json",
        )
    }
    response = client.post("/v1/files", files=files)
    assert response.status_code == 400
    assert "Invalid file format" in response.json()["detail"]

    # Test file with only whitespace
    files = {
        "file": (
            "test.jsonl",
            b"   \n  \n",
            "application/json",
        )
    }
    response = client.post("/v1/files", files=files)
    assert response.status_code == 400
    assert "Invalid file format" in response.json()["detail"]


def test_upload_file_validates_format_consistency(client):
    """Test that all records in a file must have the same format"""
    # Test file with mixed SFT and DPO formats
    mixed_content = (
        b'{"messages": [{"role": "user", "content": "test"}]}\n'
        b'{"input": {"messages": [{"role": "user", "content": "test"}], "tools": [], "parallel_tool_calls": false}, "preferred_output": [{"role": "assistant", "content": "test"}], "non_preferred_output": [{"role": "assistant", "content": "test"}]}\n'
    )
    files = {"file": ("test.jsonl", mixed_content, "application/json")}
    response = client.post("/v1/files", files=files)
    # Mixed format files should be rejected
    assert response.status_code == 400
    assert "Invalid file format" in response.json()["detail"]
