import pytest
from enum import Enum
from pydantic import BaseModel
from fastapi.testclient import TestClient
from fastapi import FastAPI
from arbor.server.api.routes.files import router
from arbor.server.services.file_manager import FileManager
from pathlib import Path

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
                {"role": "user", "content": "Who won the world series in 2020?"}
            ]
        }
    }

    response = client.post("/v1/chat/completions", json=request_json)

    print("Inference response:", response.json())

    client.app.state.inference_manager.kill()
    print("Successfully killed inference manager")
    print("Existing process:", client.app.state.inference_manager.process)
    assert client.app.state.inference_manager.process is None


class CarType(str, Enum):
    sedan = "sedan"
    suv = "SUV"
    truck = "Truck"
    coupe = "Coupe"

class CarDescription(BaseModel):
    brand: str
    model: str
    car_type: CarType

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
