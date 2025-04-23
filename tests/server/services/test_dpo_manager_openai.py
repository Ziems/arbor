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

def start():
    proc = Process(target=run_server)
    proc.start()
    time.sleep(1)
    return proc

def end(proc):
    proc.terminate()  # Shut down the server after tests
    proc.join()


def config_server():
    """Set up a test server with configured dependencies"""
    from arbor.server.main import app
    from arbor.server.core.config import Settings
    from arbor.server.services.file_manager import FileManager
    from arbor.server.services.job_manager import JobManager
    from arbor.server.services.dpo_manager import DPOManager
    from arbor.server.services.training_manager import TrainingManager

    # Use tmp_path_factory for module-scoped fixture

    # Create test settings
    settings = Settings()

    # Initialize services with test settings
    file_manager = FileManager(settings=settings)
    job_manager = JobManager(settings=settings)
    dpo_manager = DPOManager(settings=settings)
    training_manager = TrainingManager(settings=settings)

    # Inject dependencies into app state
    app.state.settings = settings
    app.state.file_manager = file_manager
    app.state.job_manager = job_manager
    app.state.dpo_manager = dpo_manager
    app.state.training_manager = training_manager

    # Start server in a separate process
    # proc = Process(target=run_server)
    # proc.start()
    # time.sleep(1)  # Give the server a moment to start
    # return app
    # proc.terminate()  # Shut down the server after tests
    # proc.join()


# @pytest.fixture(scope="module")
def trained_model_job_openai():

    from openai import OpenAI
    base_url = "http://localhost:8000/v1"

    openai_client = OpenAI(
        base_url=base_url,  # Using Arbor server
        api_key="not-needed"  # If you're using a local server, you dont need an API key
    )


    test_file_path = Path(__file__).parent.parent.parent / "data" / "training_data_dpo.jsonl"
    test_content = test_file_path.read_bytes()
    files = {"file": ("test.jsonl", test_content, "application/json")}


    # upload_response = openai_client.post("/v1/files", files=files)
    upload_response = openai_client.files.create(
        file=open(test_file_path, "rb"),
        purpose="fine-tune"
        )
    # assert upload_response.status_code == 200   #Openai doesn't have the "status_code" return
    file_id = upload_response.id

    import pdb
    dpo_response = openai_client.fine_tuning.jobs.create(
        training_file=file_id,
        model="HuggingFaceTB/SmolLM2-135M-Instruct",
        method={
            "type": "dpo",
            "dpo": {
                "hyperparameters": {"beta": 0.1},
            },
        },
    )

    print("????????????", dpo_response)

    # dpo_response = openai_client.post("/v1/fine_tuning/dpo", json={
    #     "training_file": file_id,
    #     "model": "HuggingFaceTB/SmolLM2-135M-Instruct"
    # })
    assert dpo_response.status_code == 200
    job_id = dpo_response.json()["id"]

    # 3. Poll job status until completion
    max_attempts = 30
    poll_interval = 2

    time.sleep(2)

    for _ in range(max_attempts):
        status_response = openai_client.get(f"/v1/fine_tuning/jobs/{job_id}")
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




trained_model_job_openai()

# config_server()
# proc = start()
# run_server()
# try:
#     trained_model_job_openai()
# except Exception as e:
#     print(f"An error occurred: {e}")

# end(proc)