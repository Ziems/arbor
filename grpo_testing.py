import requests
import json

def initialize_grpo(model, url='http://127.0.0.1:8000/v1/fine_tuning/grpo/initialize'):
    headers = {'Content-Type': 'application/json'}
    data = {
        'model': model,
    }
    response = requests.post(url, headers=headers, json=data)
    return response


#"HuggingFaceTB/SmolLM2-135M-Instruct"
#"Qwen/Qwen2-0.5B-Instruct"
def run_grpo_step(model_name, batch, url='http://127.0.0.1:8000/v1/fine_tuning/grpo/step'):
    headers = {'Content-Type': 'application/json'}
    data = {
        'model': model_name,
        'update_inference_model': True,
        "batch": batch
    }
    response = requests.post(url, headers=headers, json=data)
    return response

def terminate_grpo(url='http://127.0.0.1:8000/v1/fine_tuning/grpo/terminate'):
    headers = {'Content-Type': 'application/json'}
    data = {
        'status': 'success'
    }
    response = requests.post(url, headers=headers, json=data)
    return response



response = initialize_grpo(model="Qwen/Qwen2-0.5B-Instruct")
print(response.status_code)
print(response.text)

test_batch = [
        {
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather in San Francisco?"
                }
            ]
        },
        "completions": [
            {
                "role": "assistant",
                "content": "The weather in San Francisco is 70 degrees Fahrenheit.",
                "reward": 3
            },
            {
                "role": "assistant",
                "content": "The weather in San Francisco is 21 degrees Celsius.",
                "reward": 1
            },
            {
                "role": "assistant",
                "content": "As an artificial intelligence, I don't have real-time weather information and cannot give you an exact answer. To obtain the current weather in San Francisco, you can use a weather app or website like Weather.com that provides up-to-date weather conditions for a specific location.",
                "reward": 1
            }
        ]
        },
    ]
# Example usage (continuing from previous upload):
fine_tune_response = run_grpo_step(model_name="Qwen/Qwen2-0.5B-Instruct", batch=test_batch)
print(fine_tune_response.status_code)
print(fine_tune_response.text)

