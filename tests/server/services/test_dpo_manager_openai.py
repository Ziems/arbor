import time
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import pytest
from multiprocessing import Process
import uvicorn
from arbor.server.main import app

# Make sure to run "uv run arbor serve --arbor-config arbor.yaml --port 8001" in another terminal

from openai import OpenAI
base_url = "http://localhost:8001/v1"

openai_client = OpenAI(
    base_url=base_url,  # Using Arbor server
    api_key="not-needed"  # If you're using a local server, you dont need an API key
)


test_file_path = Path(__file__).parent.parent.parent / "data" / "training_data_pft.jsonl"
test_content = test_file_path.read_bytes()
files = {"file": ("test.jsonl", test_content, "application/json")}


# upload_response = openai_client.post("/v1/files", files=files)
upload_response = openai_client.files.create(
    file=open(test_file_path, "rb"),
    purpose="fine-tune"
    )
# assert upload_response.status_code == 200   #Openai doesn't have the "status_code" return
file_id = upload_response.id

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

job_id = dpo_response.id

# 3. Poll job status until completion
max_attempts = 30
poll_interval = 2

time.sleep(2)

for _ in range(max_attempts):
    status_response = openai_client.fine_tuning.jobs.retrieve(job_id)

    status = status_response.status
    if status == "succeeded":
        break
    elif status in ["failed", "cancelled"]:
        raise AssertionError(f"Job failed with status: {status}")

    time.sleep(poll_interval)
else:
    raise AssertionError(f"Job did not complete within {max_attempts * poll_interval} seconds")

# return job_id
print(f"Job {job_id} completed successfully with status: {status}")
