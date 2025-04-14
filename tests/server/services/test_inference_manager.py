import pytest
from enum import Enum
from pydantic import BaseModel
from fastapi.testclient import TestClient
from fastapi import FastAPI
from arbor.server.api.routes.files import router
from arbor.server.services.file_manager import FileManager
from pathlib import Path
from openai import OpenAI

@pytest.fixture(scope="module")
def server(tmp_path_factory):
    """Set up a test server with configured dependencies"""
    from arbor.server.main import app
    from arbor.server.core.config import Settings
    from arbor.server.services.inference_manager import InferenceManager
    from arbor.server.services.job_manager import JobManager

    # Use tmp_path_factory instead of tmp_path because we're using scope="module"
    test_storage = tmp_path_factory.mktemp("test_storage")

    # Create test settings
    settings = Settings(
        STORAGE_PATH=str(test_storage)
    )

    # Initialize services with test settings
    inference_manager = InferenceManager(settings=settings)
    job_manager = JobManager(settings=settings)
    

    # Inject dependencies into app state
    app.state.settings = settings
    app.state.inference_manager = inference_manager
    app.state.job_manager = job_manager

    return app


class CarType(str, Enum):
    sedan = "sedan"
    suv = "SUV"
    truck = "Truck"
    coupe = "Coupe"

class CarDescription(BaseModel):
    brand: str
    model: str
    car_type: CarType

# @pytest.fixture
def test_simple_inference_openai(server):
    server.state.inference_manager.launch("Qwen/Qwen2.5-1.5B-Instruct")
    host = server.state.inference_manager.host
    port = server.state.inference_manager.port
    base_url = f"http://{host}:{port}/v1"
    client = OpenAI(
        base_url=base_url,  # Using Arbor server
        api_key="not-needed"  # If you're using a local server, you dont need an API key
    )

    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi! What is the capital of the moon?"},
        ]
    )


    print("Inference response:", response.json())
    server.state.inference_manager.kill()
    print("Successfully killed inference manager")
    print("test_simple_inference_openai PASSED!!!\n\n\n")


def test_structured_inference_openai(server):
    json_schema = CarDescription.model_json_schema()

    server.state.inference_manager.launch("Qwen/Qwen2.5-1.5B-Instruct")
    host = server.state.inference_manager.host
    port = server.state.inference_manager.port
    base_url = f"http://{host}:{port}/v1"
    client = OpenAI(
        base_url=base_url,  # Using Arbor server
        api_key="not-needed"  # If you're using a local server, you dont need an API key
    )
    prompt = ("Generate a JSON with the brand, model and car_type of"
          "the most iconic car from the 90's")
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        messages=[{
            "role": "user",
            "content": prompt,
        }],
        extra_body={"guided_json": json_schema},
    )

    print("Inference response:", response.json())
    server.state.inference_manager.kill()
    print("Successfully killed inference manager")
    print("test_structured_inference_openai PASSED!!!\n\n\n")



@pytest.fixture
def client(server):
    return TestClient(server)


def test_simple_inference(client):
    request_json = {
        "headers": {
            "Content-Type": "application/json"
        },

        "pload": {
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hi! What is the capital of the moon?"}
            ]
        }
    }

    response = client.post("/v1/chat/completions", json=request_json)

    print("Inference response:", response.json())

    client.app.state.inference_manager.kill()
    print("Successfully killed inference manager")
    print("Existing process:", client.app.state.inference_manager.process)
    assert client.app.state.inference_manager.process is None
    print("test_simple_inference PASSED!!!\n\n\n")



def test_structured_inference(client):
    json_schema = CarDescription.model_json_schema()
    request_json = {
        "headers": {
            "Content-Type": "application/json"
        },

        "pload": {
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Generate a JSON with the brand, model and car_type of the most iconic car from the 90's"}
            ],
            "guided_json": json_schema
        }
    }
    

    response = client.post("/v1/chat/completions", json=request_json)
    print("Inference response:", response.json())

    client.app.state.inference_manager.kill()
    print("Successfully killed inference manager")
    print("Existing process:", client.app.state.inference_manager.process)
    assert client.app.state.inference_manager.process is None
    print("test_structured_inference PASSED!!!\n\n\n")
