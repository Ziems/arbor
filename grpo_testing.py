import requests
from openai import OpenAI
from datasets import load_dataset
client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",  # Using Arbor server
    api_key="not-needed"  # If you're using a local server, you dont need an API key
)

def initialize_grpo(model, url='http://127.0.0.1:8000/v1/fine_tuning/grpo/initialize'):
    headers = {'Content-Type': 'application/json'}
    data = {
        'model': model,
        'suffix': 'test',
        'num_generations': 8
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


def reward_func(prompts, completions):

    return [-abs(20 - len(completion)) if completion is not None else -300 for completion in completions]



dataset = load_dataset("trl-lib/tldr", split="train")
initialize_response = initialize_grpo(model="Qwen/Qwen2-0.5B-Instruct")

for i in range(len(dataset)):
    inputs = dataset[i]
    input_messages = [{"role": "user", "content": inputs["prompt"]}]
    response = client.chat.completions.create(
        model="Qwen/Qwen2-0.5B-Instruct",
        messages=input_messages,
        temperature=0.7,
        n=8
    )
    completions = [{'content': choice.message.content, 'role': choice.message.role} for choice in response.choices]
    rewards = reward_func(inputs["prompt"], [c["content"] for c in completions])
    for c, r in zip(completions, rewards):
        c["reward"] = r

    batch = [{
        "input": {
            "messages": input_messages
        },
        "completions": completions
    }]
    run_grpo_step(model_name="Qwen/Qwen2-0.5B-Instruct", batch=batch)


terminate_response = terminate_grpo()