import pytest
from arbor.server.services.inference_manager import InferenceManager
from arbor.server.core.config import Settings

@pytest.fixture(scope="module")
def inference_manager():
    settings = Settings()
    manager = InferenceManager(settings)
    yield manager
    manager.kill()  # ensure cleanup after test

def test_inference(inference_manager):
    inference_manager.launch("Qwen/Qwen2.5-1.5B-Instruct")
    response = inference_manager.run_inference("How are you today?")
    print("Inference response:", response)

    inference_manager.kill()
    print("Successfully killed inference manager")
    print("Existing process:", inference_manager.process)
    assert inference_manager.process is None
