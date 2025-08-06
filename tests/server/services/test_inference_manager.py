from enum import Enum
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from openai import OpenAI
from pydantic import BaseModel

from arbor.server.api.routes.files import router
from arbor.server.services.managers.file_manager import FileManager


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    """Set up a test server with configured dependencies"""
    from arbor.server.core.config import Config
    from arbor.server.main import app
    from arbor.server.services.managers.inference_manager import InferenceManager
    from arbor.server.services.managers.job_manager import JobManager

    # Use tmp_path_factory instead of tmp_path because we're using scope="module"
    test_storage = tmp_path_factory.mktemp("test_storage")

    # Create test config
    config = Config(storage_path=str(test_storage), gpu_ids=[])

    # Initialize services with test config
    inference_manager = InferenceManager(config=config)
    job_manager = JobManager(config=config)

    # Inject dependencies into app state
    app.state.config = config
    app.state.inference_manager = inference_manager
    app.state.job_manager = job_manager

    yield app

    # Cleanup after all tests in this module
    try:
        app.state.job_manager.cleanup()
    except Exception as e:
        print(f"Error during job manager cleanup: {e}")


@pytest.fixture(scope="module")
def client(server):
    return TestClient(server)


class CarType(str, Enum):
    sedan = "sedan"
    suv = "suv"
    hatchback = "hatchback"
    coupe = "coupe"


class CarDescription(BaseModel):
    make: str
    model: str
    year: int
    color: str
    car_type: CarType


# TODO: These inference manager tests need to be updated to match current API
# The inference manager API has changed and these tests are outdated
# def test_simple_inference_openai(server):
#     # Test code would go here
#     pass

# def test_structured_inference_openai(server):
#     # Test code would go here
#     pass

# def test_simple_inference(client):
#     # Test code would go here
#     pass

# def test_structured_inference(client):
#     # Test code would go here
#     pass


def test_inference_manager_exists(server):
    """Basic test to verify inference manager is set up correctly."""
    assert hasattr(server.state, "inference_manager")
    assert server.state.inference_manager is not None
    print("Inference manager is properly initialized")
