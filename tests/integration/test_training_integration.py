# File train manager tests - using TestClient for simplicity
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from arbor.server.main import app


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    """Set up a test server with configured dependencies using TestClient"""
    from arbor.server.core.config import Config
    from arbor.server.services.managers.file_manager import FileManager
    from arbor.server.services.managers.file_train_manager import FileTrainManager
    from arbor.server.services.managers.job_manager import JobManager

    # Use tmp_path_factory for module-scoped fixture
    test_storage = tmp_path_factory.mktemp("test_storage")

    # Create test config
    config = Config(storage_path=str(test_storage), gpu_ids=[])

    # Initialize services with test settings
    file_manager = FileManager(config=config)
    job_manager = JobManager(config=config)
    file_train_manager = FileTrainManager(config=config)

    # Inject dependencies into app state
    app.state.config = config
    app.state.file_manager = file_manager
    app.state.job_manager = job_manager
    app.state.file_train_manager = file_train_manager

    return app


@pytest.fixture(scope="module")
def client(server):
    return TestClient(server)


@pytest.fixture(scope="module")
def sample_training_file(client):
    """Upload a training file and return its ID"""
    test_file_path = Path(__file__).parent.parent / "data" / "training_data_sft.jsonl"
    test_content = test_file_path.read_bytes()
    files = {"file": ("test.jsonl", test_content, "application/json")}

    upload_response = client.post("/v1/files", files=files)
    assert upload_response.status_code == 200
    return upload_response.json()["id"]


def test_create_fine_tune_job(sample_training_file, client):
    """Test creating a fine-tune job - should be fast with ARBOR_MOCK_GPU=1"""
    finetune_response = client.post(
        "/v1/fine_tuning/jobs",
        json={
            "training_file": sample_training_file,
            "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
        },
    )
    assert finetune_response.status_code == 200
    job_data = finetune_response.json()
    print(job_data)
    assert "ftjob" in job_data["id"]
    assert "training_file" in job_data
    assert job_data["model"] == "HuggingFaceTB/SmolLM2-135M-Instruct"


def test_get_fine_tune_job(sample_training_file, client):
    """Test getting a fine-tune job status"""
    # Create job
    finetune_response = client.post(
        "/v1/fine_tuning/jobs",
        json={
            "training_file": sample_training_file,
            "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
        },
    )
    job_id = finetune_response.json()["id"]

    # Get job status
    status_response = client.get(f"/v1/fine_tuning/jobs/{job_id}")
    assert status_response.status_code == 200
    job_data = status_response.json()
    assert job_data["id"] == job_id
    assert "status" in job_data


def test_list_fine_tune_jobs(sample_training_file, client):
    """Test listing fine-tune jobs"""
    # Create a job
    client.post(
        "/v1/fine_tuning/jobs",
        json={
            "training_file": sample_training_file,
            "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
        },
    )

    # List jobs
    list_response = client.get("/v1/fine_tuning/jobs")
    assert list_response.status_code == 200
    jobs_data = list_response.json()
    assert "data" in jobs_data
    assert len(jobs_data["data"]) >= 1


# Integration test that actually runs training would go in a separate integration test file
# def test_complete_training_workflow(sample_training_file, client):
#     """This would be moved to integration tests since it requires real training"""
#     pass
