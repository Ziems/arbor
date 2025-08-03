# Assumes that the server is running
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from datasets import load_dataset
from openai import OpenAI

arbor_port = 7453

client = OpenAI(
    base_url=f"http://127.0.0.1:{arbor_port}/v1",  # Using Arbor server
    api_key="not-needed",  # If you're using a local server, you dont need an API key
)


def initialize_grpo(
    model, url=f"http://127.0.0.1:{arbor_port}/v1/fine_tuning/grpo/initialize"
):
    headers = {"Content-Type": "application/json"}
    data = {"model": model, "num_generations": 8, "lora": True, "grpo_flavor": "grpo"}
    response = requests.post(url, headers=headers, json=data)
    return response


# "HuggingFaceTB/SmolLM2-135M-Instruct"
# "Qwen/Qwen2-0.5B-Instruct"
def run_grpo_step(
    model_name, batch, url=f"http://127.0.0.1:{arbor_port}/v1/fine_tuning/grpo/step"
):
    headers = {"Content-Type": "application/json"}
    data = {"model": model_name, "batch": batch}
    response = requests.post(url, headers=headers, json=data)
    return response


def checkpoint(
    checkpoint_name, url=f"http://127.0.0.1:{arbor_port}/v1/fine_tuning/grpo/checkpoint"
):
    headers = {"Content-Type": "application/json"}
    data = {"checkpoint_name": checkpoint_name}
    response = requests.post(url, headers=headers, json=data)
    return response


def terminate_grpo(url=f"http://127.0.0.1:{arbor_port}/v1/fine_tuning/grpo/terminate"):
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers)
    return response


def main():
    def _reward_func(prompts, completions):

        return [
            -abs(20 - len(completion)) if completion is not None else -300
            for completion in completions
        ]

    def _single_chat_completion(model, messages, temperature=0.7):
        """Function to handle a single chat completion request"""
        response = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature
        )
        choice = response.choices[0]
        return {"content": choice.message.content, "role": choice.message.role}

    dataset = load_dataset("trl-lib/tldr", split="train")
    current_model = "Qwen/Qwen3-0.6B"
    initialize_response = initialize_grpo(model=current_model)
    last_checkpoint = None

    tik = time.time()
    for i in range(len(dataset)):
        inputs = dataset[i]
        input_messages = [{"role": "user", "content": inputs["prompt"]}]
        completions = []
        for _ in range(8):
            response = client.chat.completions.create(
                model=current_model, messages=input_messages, temperature=0.7
            )
            # Assuming response.choices[0] is the single completion
            choice = response.choices[0]
            completions.append(
                {"content": choice.message.content, "role": choice.message.role}
            )
        rewards = _reward_func(inputs["prompt"], [c["content"] for c in completions])
        print(rewards)

        batch = []
        for completion, reward in zip(completions, rewards):
            batch.append(
                {"messages": input_messages, "completion": completion, "reward": reward}
            )
        step_response = run_grpo_step(model_name=current_model, batch=batch)
        current_model = step_response.json()["current_model"]

        if i == 10:
            checkpoint_response = checkpoint(checkpoint_name=f"checkpoint_{i}")
            last_checkpoint_name = checkpoint_response.json()["last_checkpoint"]

        if i == 20:
            break
    tok = time.time()
    print(f"Time taken: {tok - tik} seconds")
    terminate_response = terminate_grpo()
    import pdb

    pdb.set_trace()
    inputs = dataset[-1]
    input_messages = [{"role": "user", "content": inputs["prompt"]}]
    response = client.chat.completions.create(
        model=current_model, messages=input_messages, temperature=0.7, n=8
    )


if __name__ == "__main__":
    main()
