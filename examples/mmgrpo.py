# Assumes that the server is running
import atexit
import signal
import sys

import numpy as np
import requests
from datasets import load_dataset
from openai import OpenAI

arbor_port = 7453

client = OpenAI(
    base_url=f"http://127.0.0.1:{arbor_port}/v1",  # Using Arbor server
    api_key="not-needed",  # If you're using a local server, you dont need an API key
)

# Global flag to track if GRPO is initialized
grpo_initialized = False


def cleanup_grpo():
    """Cleanup function to terminate GRPO if it was initialized"""
    global grpo_initialized
    if grpo_initialized:
        print("\nCleaning up: Terminating GRPO...")
        try:
            terminate_response = terminate_grpo()
            print(f"GRPO terminated successfully: {terminate_response.status_code}")
        except Exception as e:
            print(f"Error during GRPO cleanup: {e}")
        grpo_initialized = False


def signal_handler(signum, frame):
    """Handle interrupt signals (Ctrl+C, etc.)"""
    print(f"\nReceived signal {signum}, cleaning up...")
    cleanup_grpo()
    sys.exit(0)


# Register cleanup functions
atexit.register(cleanup_grpo)
signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination signal


def initialize_grpo(
    model, url=f"http://127.0.0.1:{arbor_port}/v1/fine_tuning/grpo/initialize"
):
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "grpo_flavor": "mmgrpo",
        "logging_steps": 10,
        "report_to": "wandb",
        "lora": True,
        "max_grad_norm": 0.1,
    }
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
    global grpo_initialized

    def _reward_func(completions, **kwargs):
        return [-abs(20 - len(completion)) for completion in completions]

    try:
        dataset = load_dataset("trl-lib/tldr", split="train")
        current_model = "Qwen/Qwen2-0.5B-Instruct"

        # Initialize GRPO
        print("Initializing GRPO...")
        initialize_response = initialize_grpo(model=current_model)
        if initialize_response.status_code != 200:
            raise Exception(f"Failed to initialize GRPO: {initialize_response.text}")
        grpo_initialized = True
        print("GRPO initialized successfully")

        last_checkpoint_name = None

        for i in range(len(dataset)):
            should_print = i % 10 == 0

            if should_print:
                print(f"Processing step {i}/{len(dataset)}")

            inputs = dataset[i]
            input_messages = [{"role": "user", "content": inputs["prompt"]}]
            responses = [
                client.chat.completions.create(
                    model=current_model, messages=input_messages, temperature=0.7, n=8
                )
                for _ in range(8)
            ]

            completions = [
                {"content": choice.message.content, "role": choice.message.role}
                for response in responses
                for choice in response.choices
            ]
            rewards = _reward_func([c["content"] for c in completions])

            if should_print:
                print(
                    f"Rewards: {[f'{r:.2f}' for r in rewards]} Avg: {np.mean(rewards):.2f} Std: {np.std(rewards):.2f}"
                )

            # Normalize rewards to calculate advantages
            rewards = np.array(rewards, dtype=np.float32)
            mean_reward = rewards.mean()
            std_reward = rewards.std()
            if std_reward > 0:
                normalized_rewards = (rewards - mean_reward) / (std_reward + 1e-4)
            else:
                normalized_rewards = rewards - mean_reward
            advantages = normalized_rewards.tolist()

            if should_print:
                print(
                    f"Advantages: {[f'{a:.2f}' for a in advantages]} Avg: {np.mean(advantages):.2f} Std: {np.std(advantages):.2f}"
                )

            batch = []
            for completion, advantage in zip(completions, advantages):
                batch.append(
                    [  # only a single module for now
                        {
                            "messages": input_messages,
                            "completion": completion,
                            "advantage": advantage,
                        }
                    ]
                )

            step_response = run_grpo_step(model_name=current_model, batch=batch)
            if step_response.status_code != 200:
                print(f"Warning: GRPO step failed: {step_response.text}")
                continue

            current_model = step_response.json()["current_model"]

            # if i == 50:
            #     print("Creating checkpoint...")
            #     checkpoint_response = checkpoint(checkpoint_name=f"checkpoint_{i}")
            #     if checkpoint_response.status_code == 200:
            #         last_checkpoint_name = checkpoint_response.json()["last_checkpoint"]
            #         print(f"Checkpoint created: {last_checkpoint_name}")
            #     else:
            #         print(f"Checkpoint failed: {checkpoint_response.text}")

        # Test final model
        print("Testing final model...")
        inputs = dataset[-1]
        input_messages = [{"role": "user", "content": inputs["prompt"]}]
        response = client.chat.completions.create(
            model=current_model, messages=input_messages, temperature=0.7, n=8
        )
        print("Training completed successfully!")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error during training: {e}")
        raise
    finally:
        # cleanup_grpo() will be called automatically via atexit
        # but we can also call it explicitly here for immediate cleanup
        if grpo_initialized:
            print("Training finished, cleaning up...")


if __name__ == "__main__":
    main()
