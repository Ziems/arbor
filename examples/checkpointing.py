import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from arbor.server.api.models.schemas import GRPOStatus
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
) -> GRPOStatus:
    headers = {"Content-Type": "application/json"}
    trainer_config = {
        "num_generations": 8,
        "temperature": 1.0,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-6,
        "beta": 0.01,
        "max_grad_norm": 1.0,
        "max_prompt_length": 512,
        "max_seq_len": 1024,
        "mask_truncated_completions": True,
        "gradient_checkpointing": True,
        "lr_scheduler_type": "constant_with_warmup",
        "warmup_steps": 10,
        # "lora_config": {
        #     "r": 16,
        #     "lora_alpha": 64,
        #     "target_modules": [
        #         "q_proj",
        #         "k_proj",
        #         "v_proj",
        #         "o_proj",
        #         "up_proj",
        #         "down_proj",
        #         "gate_proj",
        #     ],
        #     "lora_dropout": 0.05,
        # },
        "max_steps": 1000,
        "bf16": True,
        "report_to": "wandb",
        "logging_steps": 10,
        # "scale_rewards": False,
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
                "num_inference_gpus": 2,
                "num_training_gpus": 2,
            },
        },
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return GRPOStatus.model_validate(response.json())

def get_grpo_status(
    job_id,
) -> GRPOStatus:
    url = f"http://127.0.0.1:{arbor_port}/v1/fine_tuning/grpo/status"
    headers = {"Content-Type": "application/json"}
    body = {"job_id": job_id}
    response = requests.post(url, headers=headers, json=body)
    response.raise_for_status()
    res_json = response.json()
    print(res_json)
    return GRPOStatus.model_validate(res_json)

# "HuggingFaceTB/SmolLM2-135M-Instruct"
# "Qwen/Qwen2-0.5B-Instruct"
def run_grpo_step(
    model_name,
    batch,
    batch_id,
    job_id,
    url=f"http://127.0.0.1:{arbor_port}/v1/fine_tuning/grpo/step",
) -> GRPOStatus:
    headers = {"Content-Type": "application/json"}
    data = {"model": model_name, "batch": batch, "job_id": job_id, "batch_id": batch_id}
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return GRPOStatus.model_validate(response.json())


def checkpoint(
    checkpoint_name,
    job_id,
    url=f"http://127.0.0.1:{arbor_port}/v1/fine_tuning/grpo/checkpoint",
) -> GRPOStatus:
    headers = {"Content-Type": "application/json"}
    data = {"checkpoint_name": checkpoint_name, "job_id": job_id}
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return GRPOStatus.model_validate(response.json())


def terminate_grpo(
    job_id, url=f"http://127.0.0.1:{arbor_port}/v1/fine_tuning/grpo/terminate"
) -> GRPOStatus:
    headers = {"Content-Type": "application/json"}
    data = {"job_id": job_id}
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return GRPOStatus.model_validate(response.json())


def main():
    def _unique_letter_reward(completions: list[str]) -> float:
        rewards = []
        for completion in completions:
            letters = [ch.lower() for ch in completion if ch.isalpha()]
            rewards.append(float(len(set(letters))))
        return rewards

    dataset = load_dataset("trl-lib/ultrafeedback-prompt", split="train")
    # dataset = load_dataset("json", data_files="input_prompts.jsonl", split="train")
    # import pdb; pdb.set_trace()

    current_model = "Qwen/Qwen2-0.5B-Instruct"
    try:

        initialize_response = initialize_grpo(model=current_model)
        current_model = initialize_response.current_model
        job_id = initialize_response.job_id
        last_checkpoint = None

        def _create_batch_result(batch_id):
            input_messages = dataset[batch_id]["prompt"]
            # input_messages = [dataset[batch_id]]


            response = client.chat.completions.create(
                model=current_model, messages=input_messages, temperature=1.0, n=8, top_p=1.0
            )
            completions = []
            for choice in response.choices:
                completions.append(
                    {"content": choice.message.content, "role": choice.message.role}
                )
            rewards = _unique_letter_reward([c["content"] for c in completions])
            batch = []
            for completion, reward in zip(completions, rewards):
                batch.append(
                    {"messages": input_messages, "completion": completion, "reward": reward}
                )
            return batch

        pending_batch_ids = []
        fulfilled_batch_ids = []
        while len(fulfilled_batch_ids) < 100:
            status: GRPOStatus = get_grpo_status(job_id)
            pending_batch_ids = status.pending_batch_ids
            for batch_id in pending_batch_ids:
                if batch_id not in fulfilled_batch_ids:
                    batch_result = _create_batch_result(batch_id)
                    # with open(f"batch_result_simple.jsonl", "a") as f:
                        # f.write(json.dumps(batch_result) + "\n")
                    run_grpo_step(
                        model_name=current_model, batch=batch_result, job_id=job_id, batch_id=batch_id
                    )
                    fulfilled_batch_ids.append(batch_id)

            else:
                print("All batches are fulfilled")
                time.sleep(1)
        checkpoint_response = checkpoint(checkpoint_name="checkpoint_1", job_id=job_id)
        if checkpoint_response.status_code == 200:
            last_checkpoint = checkpoint_response.json()["last_checkpoint"]
            print(f"Checkpoint created: {last_checkpoint}")
        else:
            print(f"Checkpoint failed: {checkpoint_response.text}")
    except Exception as e:
        print(e)
    finally:
        arbor.shutdown()


if __name__ == "__main__":
    main()
