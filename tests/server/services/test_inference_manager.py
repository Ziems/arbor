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
    print("Launch complete!!!")

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
    
    # request_json = {
    #     headers: {"User-Agent": "Test Client"},
    #     pload: {
    #         "prompt": "How are you today",
    #         "n": 1,
    #         "temperature": 0.0,
    #         "max_tokens": 16,
    #         "stream": False,
    #     }
    # }
    response = inference_manager.run_inference(request_json)
    print("Inference response:", response.json())

    inference_manager.kill()
    print("Successfully killed inference manager")
    print("Existing process:", inference_manager.process)
    assert inference_manager.process is None
