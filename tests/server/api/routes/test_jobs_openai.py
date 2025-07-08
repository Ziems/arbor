from pathlib import Path

import pytest

from arbor.server.api.models.schemas import JobStatus

# Configurable test model - can be changed for different test scenarios
TEST_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


# OpenAI Client Tests
@pytest.fixture(scope="session")
def live_server_url(tmp_path_factory):
    """Start a real server for OpenAI client tests"""
    import socket
    import threading
    import time

    import uvicorn

    from arbor.server.core.config import (
        ArborConfig,
        InferenceConfig,
        Settings,
        TrainingConfig,
    )
    from arbor.server.main import app
    from arbor.server.services.managers.file_manager import FileManager
    from arbor.server.services.managers.file_train_manager import FileTrainManager
    from arbor.server.services.managers.inference_manager import InferenceManager
    from arbor.server.services.managers.job_manager import JobManager

    # Set up the same dependencies as the test server fixture
    test_storage = tmp_path_factory.mktemp("test_storage")
    settings = Settings(
        STORAGE_PATH=str(test_storage),
        arbor_config=ArborConfig(
            inference=InferenceConfig(gpu_ids="2"), training=TrainingConfig(gpu_ids="3")
        ),
    )

    # Configure app state
    app.state.file_manager = FileManager(settings)
    app.state.file_train_manager = FileTrainManager(settings)
    app.state.job_manager = JobManager(settings)
    app.state.inference_manager = InferenceManager(settings)
    app.state.log_dir = str(test_storage / "logs")

    # Find a free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]

    # Start server in background thread
    server_thread = threading.Thread(
        target=uvicorn.run,
        args=(app,),
        kwargs={
            "host": "127.0.0.1",
            "port": port,
            "log_level": "error",  # Suppress logs during testing
        },
        daemon=True,
    )
    server_thread.start()

    # Wait for server to start
    time.sleep(2)

    return f"http://127.0.0.1:{port}"


@pytest.fixture
def openai_client(live_server_url):
    """Create an OpenAI client that talks to the live test server"""
    try:
        import openai

        return openai.OpenAI(api_key="test-key", base_url=f"{live_server_url}/v1")
    except ImportError:
        pytest.skip("OpenAI package not installed")


@pytest.fixture
def sample_file_sft_live(live_server_url):
    """Create a sample file for testing with the live server"""
    import httpx

    # Load the test file from tests/data
    test_file_path = (
        Path(__file__).parent.parent.parent.parent / "data" / "training_data_sft.jsonl"
    )
    valid_content = test_file_path.read_bytes()

    # Upload file to the live server
    files = {"file": ("test.jsonl", valid_content, "application/json")}
    response = httpx.post(f"{live_server_url}/v1/files", files=files)
    assert response.status_code == 200
    return response.json()["id"]


def test_openai_create_fine_tune_job(openai_client, sample_file_sft_live):
    """Test creating a fine-tune job using OpenAI client"""
    try:
        # Create fine-tuning job using OpenAI client
        job = openai_client.fine_tuning.jobs.create(
            model=TEST_MODEL, training_file=sample_file_sft_live, suffix="test_openai"
        )
        print(job)

        # Verify the response structure
        assert hasattr(job, "id")
        assert hasattr(job, "status")
        assert hasattr(job, "object")
        assert job.object == "fine_tuning.job"
        assert "ftjob" in job.id
        assert job.status == JobStatus.QUEUED.value

    except Exception as e:
        # If the OpenAI client doesn't support fine-tuning yet, skip the test
        if "fine_tuning" not in str(e):
            raise
        pytest.skip("OpenAI client doesn't support fine-tuning endpoints")


def test_openai_list_fine_tune_jobs(openai_client, sample_file_sft_live):
    """Test listing fine-tune jobs using OpenAI client"""
    try:
        # Create a job first
        openai_client.fine_tuning.jobs.create(
            model=TEST_MODEL, training_file=sample_file_sft_live, suffix="test_list"
        )

        # List jobs
        jobs = openai_client.fine_tuning.jobs.list()
        print(jobs)

        # Verify the response structure
        assert hasattr(jobs, "data")
        assert hasattr(jobs, "object")
        assert jobs.object == "list"
        assert isinstance(jobs.data, list)

        # Should have at least one job
        assert len(jobs.data) >= 1

        if jobs.data:
            job = jobs.data[0]
            assert hasattr(job, "id")
            assert hasattr(job, "status")
            assert hasattr(job, "object")

    except Exception as e:
        if "fine_tuning" not in str(e):
            raise
        pytest.skip("OpenAI client doesn't support fine-tuning endpoints")


def test_openai_get_fine_tune_job(openai_client, sample_file_sft_live):
    """Test getting a specific fine-tune job using OpenAI client"""
    try:
        # Create a job first
        created_job = openai_client.fine_tuning.jobs.create(
            model=TEST_MODEL, training_file=sample_file_sft_live, suffix="test_get"
        )

        # Get the job
        job = openai_client.fine_tuning.jobs.retrieve(created_job.id)
        print("Retrieved job:", job)

        # Verify the response structure
        assert job.id == created_job.id
        assert hasattr(job, "status")
        assert hasattr(job, "object")
        assert job.object == "fine_tuning.job"

    except Exception as e:
        if "fine_tuning" not in str(e):
            raise
        pytest.skip("OpenAI client doesn't support fine-tuning endpoints")


def test_openai_cancel_fine_tune_job(openai_client, sample_file_sft_live):
    """Test canceling a fine-tune job using OpenAI client"""
    try:
        # Create a job first
        job = openai_client.fine_tuning.jobs.create(
            model=TEST_MODEL, training_file=sample_file_sft_live, suffix="test_cancel"
        )

        # Cancel the job
        canceled_job = openai_client.fine_tuning.jobs.cancel(job.id)

        # Verify the response structure
        assert canceled_job.id == job.id
        assert hasattr(canceled_job, "status")
        assert hasattr(canceled_job, "object")
        assert canceled_job.object == "fine_tuning.job"
        assert canceled_job.status == JobStatus.PENDING_CANCEL.value

    except Exception as e:
        if "fine_tuning" not in str(e):
            raise
        pytest.skip("OpenAI client doesn't support fine-tuning endpoints")


def test_openai_list_fine_tune_events(openai_client, sample_file_sft_live):
    """Test listing fine-tune events using OpenAI client"""
    try:
        # Create a job first
        job = openai_client.fine_tuning.jobs.create(
            model=TEST_MODEL, training_file=sample_file_sft_live, suffix="test_events"
        )

        # List events
        events = openai_client.fine_tuning.jobs.list_events(job.id)

        # Verify the response structure
        assert hasattr(events, "data")
        assert hasattr(events, "object")
        assert events.object == "list"
        assert isinstance(events.data, list)

        # Events should be a list (may be empty for new jobs)
        for event in events.data:
            assert hasattr(event, "id")
            assert hasattr(event, "level")
            assert hasattr(event, "message")
            assert hasattr(event, "data")
            assert hasattr(event, "created_at")
            assert hasattr(event, "type")

    except Exception as e:
        if "fine_tuning" not in str(e):
            raise
        pytest.skip("OpenAI client doesn't support fine-tuning endpoints")


def test_openai_fine_tune_job_with_hyperparameters(openai_client, sample_file_sft_live):
    """Test creating a fine-tune job with hyperparameters using OpenAI client"""
    try:
        # Create fine-tuning job with hyperparameters
        job = openai_client.fine_tuning.jobs.create(
            model=TEST_MODEL,
            training_file=sample_file_sft_live,
            hyperparameters={
                "n_epochs": 1,
                "batch_size": 1,
                "learning_rate_multiplier": 1.0,
            },
            suffix="test_hyperparams",
        )

        # Verify the response structure
        assert hasattr(job, "id")
        assert hasattr(job, "status")
        assert hasattr(job, "object")
        assert job.object == "fine_tuning.job"
        assert "ftjob" in job.id
        assert job.status == JobStatus.QUEUED.value

    except Exception as e:
        if "fine_tuning" not in str(e):
            raise
        pytest.skip("OpenAI client doesn't support fine-tuning endpoints")


def test_openai_fine_tune_job_with_integrations(openai_client, sample_file_sft_live):
    """Test creating a fine-tune job with integrations using OpenAI client"""
    try:
        # Create fine-tuning job with integrations
        job = openai_client.fine_tuning.jobs.create(
            model=TEST_MODEL,
            training_file=sample_file_sft_live,
            integrations=[
                {
                    "type": "wandb",
                    "wandb": {
                        "project": "test_project",
                        "name": "test_run",
                        "entity": "test_entity",
                        "tags": ["test", "fine_tuning"],
                    },
                }
            ],
            suffix="test_integrations",
        )

        # Verify the response structure
        assert hasattr(job, "id")
        assert hasattr(job, "status")
        assert hasattr(job, "object")
        assert job.object == "fine_tuning.job"
        assert "ftjob" in job.id
        assert job.status == JobStatus.QUEUED.value

    except Exception as e:
        if "fine_tuning" not in str(e):
            raise
        pytest.skip("OpenAI client doesn't support fine-tuning endpoints")


def test_openai_inference(openai_client):
    """Test inference using OpenAI client"""
    response = openai_client.chat.completions.create(
        model=TEST_MODEL, messages=[{"role": "user", "content": "Hello, how are you?"}]
    )
    print(response)
    assert response.choices[0].message.content is not None
