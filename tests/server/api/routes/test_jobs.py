import pytest
from pathlib import Path

from arbor.server.api.models.schemas import JobStatus
from arbor.server.core.config import Settings, ArborConfig, InferenceConfig, TrainingConfig

# Configurable test model - can be changed for different test scenarios
TEST_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    """Set up a test server with configured dependencies"""
    from arbor.server.core.config import Settings, ArborConfig, InferenceConfig, TrainingConfig
    from arbor.server.main import app
    from arbor.server.services.managers.file_manager import FileManager
    from arbor.server.services.managers.file_train_manager import FileTrainManager
    from arbor.server.services.managers.job_manager import JobManager

    # Use tmp_path_factory instead of tmp_path because we're using scope="module"
    test_storage = tmp_path_factory.mktemp("test_storage")

    # Create test settings with required arbor_config
    settings = Settings(
        STORAGE_PATH=str(test_storage),
        arbor_config=ArborConfig(
            inference=InferenceConfig(),
            training=TrainingConfig()
        )
    )

    # Set up dependencies
    app.state.file_manager = FileManager(settings)
    app.state.file_train_manager = FileTrainManager(settings)
    app.state.job_manager = JobManager(settings)

    return app


@pytest.fixture
def client(server):
    from fastapi.testclient import TestClient
    return TestClient(server)


@pytest.fixture
def sample_file(client):
    """Create a sample file for testing"""
    # Load the test file from tests/data
    test_file_path = (
        Path(__file__).parent.parent.parent.parent / "data" / "training_data_sft.jsonl"
    )
    valid_content = test_file_path.read_bytes()

    # Upload file
    files = {"file": ("test.jsonl", valid_content, "application/json")}
    response = client.post("/v1/files", files=files)
    assert response.status_code == 200
    return response.json()["id"]


def test_create_fine_tune_job(client, sample_file):
    """Test creating a fine-tune job"""
    fine_tune_request = {
        "model": TEST_MODEL,
        "training_file": sample_file,
        "method": {
            "type": "supervised",
            "supervised": {
                "hyperparameters": {
                    "batch_size": 1,
                    "learning_rate_multiplier": 1.0,
                    "n_epochs": 1
                }
            }
        },
        "suffix": "test"
    }

    response = client.post("/v1/jobs", json=fine_tune_request)
    assert response.status_code == 200
    
    data = response.json()
    assert data["object"] == "fine_tuning.job"
    assert "ftjob" in data["id"]
    assert data["status"] == JobStatus.QUEUED.value


def test_create_fine_tune_job_dpo(client, sample_file):
    """Test creating a DPO fine-tune job"""
    fine_tune_request = {
        "model": TEST_MODEL,
        "training_file": sample_file,
        "method": {
            "type": "dpo",
            "dpo": {
                "hyperparameters": {
                    "beta": 0.1,
                    "batch_size": 1,
                    "learning_rate_multiplier": 1.0,
                    "n_epochs": 1
                }
            }
        },
        "suffix": "test_dpo"
    }

    response = client.post("/v1/jobs", json=fine_tune_request)
    assert response.status_code == 200
    
    data = response.json()
    assert data["object"] == "fine_tuning.job"
    assert "ftjob" in data["id"]
    assert data["status"] == JobStatus.QUEUED.value


def test_create_fine_tune_job_invalid_file(client):
    """Test creating a fine-tune job with invalid file"""
    fine_tune_request = {
        "model": TEST_MODEL,
        "training_file": "nonexistent_file_id",
        "method": {
            "type": "supervised",
            "supervised": {
                "hyperparameters": {
                    "batch_size": 1,
                    "learning_rate_multiplier": 1.0,
                    "n_epochs": 1
                }
            }
        }
    }

    response = client.post("/v1/jobs", json=fine_tune_request)
    # This should fail because the file doesn't exist
    assert response.status_code in [400, 404, 500]


def test_get_jobs(client, sample_file):
    """Test getting list of jobs"""
    # Create a job first
    fine_tune_request = {
        "model": TEST_MODEL,
        "training_file": sample_file,
        "method": {
            "type": "supervised",
            "supervised": {
                "hyperparameters": {
                    "batch_size": 1,
                    "learning_rate_multiplier": 1.0,
                    "n_epochs": 1
                }
            }
        }
    }
    client.post("/v1/jobs", json=fine_tune_request)

    # Get jobs
    response = client.get("/v1/jobs")
    assert response.status_code == 200
    
    data = response.json()
    assert data["object"] == "list"
    assert "data" in data
    assert "has_more" in data
    assert isinstance(data["data"], list)
    
    # Should have at least one job
    assert len(data["data"]) >= 1
    if data["data"]:
        job = data["data"][0]
        assert "id" in job
        assert "status" in job
        assert "object" in job


def test_get_job_status(client, sample_file):
    """Test getting a specific job status"""
    # Create a job first
    fine_tune_request = {
        "model": TEST_MODEL,
        "training_file": sample_file,
        "method": {
            "type": "supervised",
            "supervised": {
                "hyperparameters": {
                    "batch_size": 1,
                    "learning_rate_multiplier": 1.0,
                    "n_epochs": 1
                }
            }
        }
    }
    create_response = client.post("/v1/jobs", json=fine_tune_request)
    assert create_response.status_code == 200
    job_id = create_response.json()["id"]

    # Get job status
    response = client.get(f"/v1/jobs/{job_id}")
    assert response.status_code == 200
    
    data = response.json()
    assert data["object"] == "fine_tuning.job"
    assert data["id"] == job_id
    assert "status" in data
    assert data["fine_tuned_model"] is None  # Should be None for new jobs


def test_get_job_status_not_found(client):
    """Test getting status of non-existent job"""
    response = client.get("/v1/jobs/nonexistent-job-id")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_get_job_events(client, sample_file):
    """Test getting job events"""
    # Create a job first
    fine_tune_request = {
        "model": TEST_MODEL,
        "training_file": sample_file,
        "method": {
            "type": "supervised",
            "supervised": {
                "hyperparameters": {
                    "batch_size": 1,
                    "learning_rate_multiplier": 1.0,
                    "n_epochs": 1
                }
            }
        }
    }
    create_response = client.post("/v1/jobs", json=fine_tune_request)
    assert create_response.status_code == 200
    job_id = create_response.json()["id"]

    # Get job events
    response = client.get(f"/v1/jobs/{job_id}/events")
    assert response.status_code == 200
    
    data = response.json()
    assert data["object"] == "list"
    assert "data" in data
    assert "has_more" in data
    assert isinstance(data["data"], list)
    
    # Events should be a list (may be empty for new jobs)
    for event in data["data"]:
        assert "id" in event
        assert "level" in event
        assert "message" in event
        assert "data" in event
        assert "created_at" in event
        assert "type" in event


def test_get_job_events_not_found(client):
    """Test getting events of non-existent job"""
    response = client.get("/v1/jobs/nonexistent-job-id/events")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_get_job_checkpoints(client, sample_file):
    """Test getting job checkpoints"""
    # Create a job first
    fine_tune_request = {
        "model": TEST_MODEL,
        "training_file": sample_file,
        "method": {
            "type": "supervised",
            "supervised": {
                "hyperparameters": {
                    "batch_size": 1,
                    "learning_rate_multiplier": 1.0,
                    "n_epochs": 1
                }
            }
        }
    }
    create_response = client.post("/v1/jobs", json=fine_tune_request)
    assert create_response.status_code == 200
    job_id = create_response.json()["id"]

    # Get job checkpoints
    response = client.get(f"/v1/jobs/{job_id}/checkpoints")
    assert response.status_code == 200
    
    data = response.json()
    assert data["object"] == "list"
    assert "data" in data
    assert "has_more" in data
    assert isinstance(data["data"], list)
    
    # Checkpoints should be a list (may be empty for new jobs)
    for checkpoint in data["data"]:
        assert "id" in checkpoint
        assert "fine_tuned_model_checkpoint" in checkpoint
        assert "fine_tuning_job_id" in checkpoint
        assert "metrics" in checkpoint
        assert "step_number" in checkpoint


def test_get_job_checkpoints_not_found(client):
    """Test getting checkpoints of non-existent job"""
    response = client.get("/v1/jobs/nonexistent-job-id/checkpoints")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_cancel_job(client, sample_file):
    """Test canceling a job"""
    # Create a job first
    fine_tune_request = {
        "model": TEST_MODEL,
        "training_file": sample_file,
        "method": {
            "type": "supervised",
            "supervised": {
                "hyperparameters": {
                    "batch_size": 1,
                    "learning_rate_multiplier": 1.0,
                    "n_epochs": 1
                }
            }
        }
    }
    create_response = client.post("/v1/jobs", json=fine_tune_request)
    assert create_response.status_code == 200
    job_id = create_response.json()["id"]

    # Cancel the job
    response = client.post(f"/v1/jobs/{job_id}/cancel")
    assert response.status_code == 200
    
    data = response.json()
    assert data["object"] == "fine_tuning.job"
    assert data["id"] == job_id
    assert data["status"] == JobStatus.PENDING_CANCEL.value


def test_cancel_job_not_found(client):
    """Test canceling a non-existent job"""
    response = client.post("/v1/jobs/nonexistent-job-id/cancel")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_cancel_job_already_finished(client, sample_file):
    """Test canceling a job that's already finished"""
    # Create a job first
    fine_tune_request = {
        "model": TEST_MODEL,
        "training_file": sample_file,
        "method": {
            "type": "supervised",
            "supervised": {
                "hyperparameters": {
                    "batch_size": 1,
                    "learning_rate_multiplier": 1.0,
                    "n_epochs": 1
                }
            }
        }
    }
    create_response = client.post("/v1/jobs", json=fine_tune_request)
    assert create_response.status_code == 200
    job_id = create_response.json()["id"]

    # Manually set job status to SUCCEEDED (this would normally be done by the training process)
    job_manager = client.app.state.job_manager
    job = job_manager.get_job(job_id)
    job.status = JobStatus.SUCCEEDED

    # Try to cancel the finished job
    response = client.post(f"/v1/jobs/{job_id}/cancel")
    assert response.status_code == 400
    assert "Cannot cancel job" in response.json()["detail"]


def test_create_job_minimal_request(client, sample_file):
    """Test creating a job with minimal request"""
    fine_tune_request = {
        "model": TEST_MODEL,
        "training_file": sample_file
    }

    response = client.post("/v1/jobs", json=fine_tune_request)
    assert response.status_code == 200
    
    data = response.json()
    assert data["object"] == "fine_tuning.job"
    assert "ftjob" in data["id"]
    assert data["status"] == JobStatus.QUEUED.value


def test_create_job_with_integrations(client, sample_file):
    """Test creating a job with integrations"""
    fine_tune_request = {
        "model": TEST_MODEL,
        "training_file": sample_file,
        "integrations": [
            {
                "type": "wandb",
                "wandb": {
                    "project": "test_project",
                    "name": "test_run",
                    "entity": "test_entity",
                    "tags": ["test", "fine_tuning"]
                }
            }
        ]
    }

    response = client.post("/v1/jobs", json=fine_tune_request)
    assert response.status_code == 200
    
    data = response.json()
    assert data["object"] == "fine_tuning.job"
    assert "ftjob" in data["id"]
    assert data["status"] == JobStatus.QUEUED.value


# OpenAI Client Tests
@pytest.fixture
def openai_client(server):
    """Create an OpenAI client configured to use our test server"""
    try:
        import openai
        from fastapi.testclient import TestClient
        
        # Create test client
        test_client = TestClient(server)
        
        # Configure OpenAI client to use our test server
        client = openai.OpenAI(
            api_key="test-key",  # Not used but required
            base_url="http://testserver"  # This will be intercepted by the test client
        )
        
        # Monkey patch the client to use our test client
        client._client = test_client
        
        return client
    except ImportError:
        pytest.skip("OpenAI package not installed")


def test_openai_create_fine_tune_job(openai_client, sample_file):
    """Test creating a fine-tune job using OpenAI client"""
    try:
        # Create fine-tuning job using OpenAI client
        job = openai_client.fine_tuning.jobs.create(
            model=TEST_MODEL,
            training_file=sample_file,
            suffix="test_openai"
        )
        
        # Verify the response structure
        assert hasattr(job, 'id')
        assert hasattr(job, 'status')
        assert hasattr(job, 'object')
        assert job.object == "fine_tuning.job"
        assert "ftjob" in job.id
        assert job.status == JobStatus.QUEUED.value
        
    except Exception as e:
        # If the OpenAI client doesn't support fine-tuning yet, skip the test
        if "fine_tuning" not in str(e):
            raise
        pytest.skip("OpenAI client doesn't support fine-tuning endpoints")


def test_openai_list_fine_tune_jobs(openai_client, sample_file):
    """Test listing fine-tune jobs using OpenAI client"""
    try:
        # Create a job first
        openai_client.fine_tuning.jobs.create(
            model=TEST_MODEL,
            training_file=sample_file,
            suffix="test_list"
        )
        
        # List jobs
        jobs = openai_client.fine_tuning.jobs.list()
        
        # Verify the response structure
        assert hasattr(jobs, 'data')
        assert hasattr(jobs, 'object')
        assert jobs.object == "list"
        assert isinstance(jobs.data, list)
        
        # Should have at least one job
        assert len(jobs.data) >= 1
        
        if jobs.data:
            job = jobs.data[0]
            assert hasattr(job, 'id')
            assert hasattr(job, 'status')
            assert hasattr(job, 'object')
            
    except Exception as e:
        if "fine_tuning" not in str(e):
            raise
        pytest.skip("OpenAI client doesn't support fine-tuning endpoints")


def test_openai_get_fine_tune_job(openai_client, sample_file):
    """Test getting a specific fine-tune job using OpenAI client"""
    try:
        # Create a job first
        created_job = openai_client.fine_tuning.jobs.create(
            model=TEST_MODEL,
            training_file=sample_file,
            suffix="test_get"
        )
        
        # Get the job
        job = openai_client.fine_tuning.jobs.retrieve(created_job.id)
        
        # Verify the response structure
        assert job.id == created_job.id
        assert hasattr(job, 'status')
        assert hasattr(job, 'object')
        assert job.object == "fine_tuning.job"
        
    except Exception as e:
        if "fine_tuning" not in str(e):
            raise
        pytest.skip("OpenAI client doesn't support fine-tuning endpoints")


def test_openai_cancel_fine_tune_job(openai_client, sample_file):
    """Test canceling a fine-tune job using OpenAI client"""
    try:
        # Create a job first
        job = openai_client.fine_tuning.jobs.create(
            model=TEST_MODEL,
            training_file=sample_file,
            suffix="test_cancel"
        )
        
        # Cancel the job
        canceled_job = openai_client.fine_tuning.jobs.cancel(job.id)
        
        # Verify the response structure
        assert canceled_job.id == job.id
        assert hasattr(canceled_job, 'status')
        assert hasattr(canceled_job, 'object')
        assert canceled_job.object == "fine_tuning.job"
        assert canceled_job.status == JobStatus.PENDING_CANCEL.value
        
    except Exception as e:
        if "fine_tuning" not in str(e):
            raise
        pytest.skip("OpenAI client doesn't support fine-tuning endpoints")


def test_openai_list_fine_tune_events(openai_client, sample_file):
    """Test listing fine-tune events using OpenAI client"""
    try:
        # Create a job first
        job = openai_client.fine_tuning.jobs.create(
            model=TEST_MODEL,
            training_file=sample_file,
            suffix="test_events"
        )
        
        # List events
        events = openai_client.fine_tuning.jobs.list_events(job_id=job.id)
        
        # Verify the response structure
        assert hasattr(events, 'data')
        assert hasattr(events, 'object')
        assert events.object == "list"
        assert isinstance(events.data, list)
        
        # Events should be a list (may be empty for new jobs)
        for event in events.data:
            assert hasattr(event, 'id')
            assert hasattr(event, 'level')
            assert hasattr(event, 'message')
            assert hasattr(event, 'data')
            assert hasattr(event, 'created_at')
            assert hasattr(event, 'type')
            
    except Exception as e:
        if "fine_tuning" not in str(e):
            raise
        pytest.skip("OpenAI client doesn't support fine-tuning endpoints")


def test_openai_fine_tune_job_with_hyperparameters(openai_client, sample_file):
    """Test creating a fine-tune job with hyperparameters using OpenAI client"""
    try:
        # Create fine-tuning job with hyperparameters
        job = openai_client.fine_tuning.jobs.create(
            model=TEST_MODEL,
            training_file=sample_file,
            hyperparameters={
                "n_epochs": 1,
                "batch_size": 1,
                "learning_rate_multiplier": 1.0
            },
            suffix="test_hyperparams"
        )
        
        # Verify the response structure
        assert hasattr(job, 'id')
        assert hasattr(job, 'status')
        assert hasattr(job, 'object')
        assert job.object == "fine_tuning.job"
        assert "ftjob" in job.id
        assert job.status == JobStatus.QUEUED.value
        
    except Exception as e:
        if "fine_tuning" not in str(e):
            raise
        pytest.skip("OpenAI client doesn't support fine-tuning endpoints")


def test_openai_fine_tune_job_with_integrations(openai_client, sample_file):
    """Test creating a fine-tune job with integrations using OpenAI client"""
    try:
        # Create fine-tuning job with integrations
        job = openai_client.fine_tuning.jobs.create(
            model=TEST_MODEL,
            training_file=sample_file,
            integrations=[
                {
                    "type": "wandb",
                    "wandb": {
                        "project": "test_project",
                        "name": "test_run",
                        "entity": "test_entity",
                        "tags": ["test", "fine_tuning"]
                    }
                }
            ],
            suffix="test_integrations"
        )
        
        # Verify the response structure
        assert hasattr(job, 'id')
        assert hasattr(job, 'status')
        assert hasattr(job, 'object')
        assert job.object == "fine_tuning.job"
        assert "ftjob" in job.id
        assert job.status == JobStatus.QUEUED.value
        
    except Exception as e:
        if "fine_tuning" not in str(e):
            raise
        pytest.skip("OpenAI client doesn't support fine-tuning endpoints")
