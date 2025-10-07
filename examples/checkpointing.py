# Assumes that the server is running
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from datasets import load_dataset
from openai import OpenAI

import arbor

# Start Arbor server (starts in background)
arbor_server_info = arbor.init()

arbor_port = arbor_server_info["port"]

client = OpenAI(
    base_url=f"http://127.0.0.1:{arbor_port}/v1",  # Using Arbor server
    api_key="not-needed",  # If you're using a local server, you dont need an API key
)


def initialize_grpo(
    model, url=f"http://127.0.0.1:{arbor_port}/v1/fine_tuning/grpo/initialize"
):
    headers = {"Content-Type": "application/json"}
    trainer_config = {
        "num_generations": 8,
        "temperature": 0.7,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-5,
        "beta": 0.001,
        "max_grad_norm": 1.0,
        "max_prompt_length": 512,
        "max_seq_len": 1024,
        "mask_truncated_completions": True,
        "lora_config": {
            "r": 16,
            "lora_alpha": 64,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "up_proj",
                "down_proj",
                "gate_proj",
            ],
            "lora_dropout": 0.05,
        },
        "max_steps": 1000,
        "bf16": True,
    }

    data = {
        "run_name": "checkpointing-demo",
        "model": model,
        "trainer_config": trainer_config,
        "inference_config": {
            "model": model,
            "max_context_length": 2048,
        },
        "gpu_config": {
            "type": "multi",
            "multi": {
                "num_inference_gpus": 1,
                "num_training_gpus": 1,
            },
        },
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response


# "HuggingFaceTB/SmolLM2-135M-Instruct"
# "Qwen/Qwen2-0.5B-Instruct"
def run_grpo_step(
    model_name,
    batch,
    job_id,
    url=f"http://127.0.0.1:{arbor_port}/v1/fine_tuning/grpo/step",
):
    headers = {"Content-Type": "application/json"}
    data = {"model": model_name, "batch": batch, "job_id": job_id}
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response


def checkpoint(
    checkpoint_name,
    job_id,
    url=f"http://127.0.0.1:{arbor_port}/v1/fine_tuning/grpo/checkpoint",
):
    headers = {"Content-Type": "application/json"}
    data = {"checkpoint_name": checkpoint_name, "job_id": job_id}
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response


def terminate_grpo(
    job_id, url=f"http://127.0.0.1:{arbor_port}/v1/fine_tuning/grpo/terminate"
):
    headers = {"Content-Type": "application/json"}
    data = {"job_id": job_id}
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response


def main():
    def _unique_letter_reward(completions: list[str]) -> float:
        rewards = []
        for completion in completions:
            letters = [ch.lower() for ch in completion if ch.isalpha()]
            rewards.append(float(len(set(letters))))
        return rewards

    def _single_chat_completion(model, messages, temperature=0.7):
        """Function to handle a single chat completion request"""
        response = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature
        )
        choice = response.choices[0]
        return {"content": choice.message.content, "role": choice.message.role}

    dataset = load_dataset("trl-lib/ultrafeedback-prompt", split="train")

    current_model = "Qwen/Qwen3-0.6B"
    initialize_response = initialize_grpo(model=current_model)
    current_model = initialize_response.json()["current_model"]
    job_id = initialize_response.json()["job_id"]
    last_checkpoint = None

    tik = time.time()
    for i in range(len(dataset)):
        inputs = dataset[i]
        input_messages = inputs['prompt']
        # input_messages = [{"role": "user", "content": ["prompt"]}]

        completions = []
        response = client.chat.completions.create(
            model=current_model, messages=input_messages, temperature=0.7, n=8
        )
        for choice in response.choices:
            completions.append(
                {"content": choice.message.content, "role": choice.message.role}
            )
        rewards = _unique_letter_reward([c["content"] for c in completions])
        print(rewards, sum(rewards) / len(rewards))

        batch = []
        for completion, reward in zip(completions, rewards):
            batch.append(
                {"messages": input_messages, "completion": completion, "reward": reward}
            )
        step_response = run_grpo_step(
            model_name=current_model, batch=batch, job_id=job_id
        )

        # if i == 10:
        # checkpoint_response = checkpoint(checkpoint_name=f"checkpoint_{i}", job_id=job_id)
        # last_checkpoint_name = checkpoint_response.json()["last_checkpoint"]

        if i == 200:
            break
    tok = time.time()
    print(f"Time taken: {tok - tik} seconds")
    terminate_response = terminate_grpo(job_id=job_id)

    inputs = dataset[-1]
    input_messages = [{"role": "user", "content": inputs["prompt"]}]
    response = client.chat.completions.create(
        model=current_model, messages=input_messages, temperature=0.7, n=8
    )


if __name__ == "__main__":
    main()
