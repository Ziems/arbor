# Overall test
import time
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import pytest
from multiprocessing import Process
import uvicorn
from arbor.server.main import app

def run_server():
    uvicorn.run(app, host="127.0.0.1", port=8000)

@pytest.fixture(scope="module")
def server(tmp_path_factory):
    """Set up a test server with configured dependencies"""
    from arbor.server.main import app
    from arbor.server.core.config import Settings
    from arbor.server.services.file_manager import FileManager
    from arbor.server.services.job_manager import JobManager
    from arbor.server.services.training_manager import TrainingManager

    # Use tmp_path_factory for module-scoped fixture
    test_storage = tmp_path_factory.mktemp("test_storage")

    # Create test settings
    settings = Settings(
        STORAGE_PATH=str(test_storage)
    )

    # Initialize services with test settings
    file_manager = FileManager(settings=settings)
    job_manager = JobManager(settings=settings)
    training_manager = TrainingManager(settings=settings)

    # Inject dependencies into app state
    app.state.settings = settings
    app.state.file_manager = file_manager
    app.state.job_manager = job_manager
    app.state.training_manager = training_manager

    # Start server in a separate process
    proc = Process(target=run_server)
    proc.start()
    time.sleep(1)  # Give the server a moment to start
    yield app
    proc.terminate()  # Shut down the server after tests
    proc.join()

class APIClient:
    def __init__(self, base_url):
        self.base_url = base_url
        self.session = requests.Session()

    def _url(self, path):
        return f"{self.base_url}{path}"

    def get(self, path, **kwargs):
        return self.session.get(self._url(path), **kwargs)

    def post(self, path, **kwargs):
        return self.session.post(self._url(path), **kwargs)

@pytest.fixture(scope="module")
def client():
    # Using requests for a real HTTP client
    base_url = "http://localhost:8000"
    return APIClient(base_url)

@pytest.fixture(scope="module")
def trained_model_job(server, client):
    """Fixture that runs the fine-tuning once and returns the job_id for other tests to use"""
    # 1. Upload training file
    test_file_path = Path(__file__).parent.parent.parent / "data" / "training_data_sft.jsonl"
    test_content = test_file_path.read_bytes()
    files = {"file": ("test.jsonl", test_content, "application/json")}

    upload_response = client.post("/api/files", files=files)
    assert upload_response.status_code == 200
    file_id = upload_response.json()["id"]

    # 2. Start fine-tuning job
    finetune_response = client.post("/api/fine-tune", json={
        "training_file": file_id,
        "model": "HuggingFaceTB/SmolLM2-135M-Instruct"
    })
    assert finetune_response.status_code == 200
    job_id = finetune_response.json()["id"]

    # 3. Poll job status until completion
    max_attempts = 30
    poll_interval = 2

    time.sleep(2)

    for _ in range(max_attempts):
        status_response = client.get(f"/api/job/{job_id}")
        assert status_response.status_code == 200

        status = status_response.json()["status"]
        if status == "succeeded":
            break
        elif status in ["failed", "cancelled"]:
            raise AssertionError(f"Job failed with status: {status}")

        time.sleep(poll_interval)
    else:
        raise AssertionError(f"Job did not complete within {max_attempts * poll_interval} seconds")

    return job_id

def test_complete_workflow(trained_model_job, client):
    # Just verify final status since training is already done
    final_response = client.get(f"/api/job/{trained_model_job}")
    assert final_response.status_code == 200
    assert final_response.json()["status"] == "succeeded"
    assert final_response.json()["fine_tuned_model"] is not None

def test_model_can_be_loaded(trained_model_job, client):
    # Your test code here, using trained_model_job
    final_response = client.get(f"/api/job/{trained_model_job}")
    model_path = final_response.json()["fine_tuned_model"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not model_path:
        raise ValueError("No fine-tuned model path found in job response")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    assert model is not None
    assert tokenizer is not None

def test_model_can_be_prompted(trained_model_job, client):
    # Your test code here, using trained_model_job
    final_response = client.get(f"/api/job/{trained_model_job}")
    model_path = final_response.json()["fine_tuned_model"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not model_path:
        raise ValueError("No fine-tuned model path found in job response")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    prompt = "Hello, how are you?"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    assert len(outputs) > 0
    assert outputs[0].shape[0] > 0
    assert outputs[0].shape[0] > 0
    text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assert text_output is not None