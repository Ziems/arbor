# Assumes that the server is running
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
    data = {"model": model, "num_generations": 8, "update_interval": 5}
    response = requests.post(url, headers=headers, json=data)
    return response


# "HuggingFaceTB/SmolLM2-135M-Instruct"
# "Qwen/Qwen2-0.5B-Instruct"
def run_grpo_step(
    model_name, batch, url=f"http://127.0.0.1:{arbor_port}/v1/fine_tuning/grpo/step"
):
    headers = {"Content-Type": "application/json"}
    data = {"model": model_name, "update_inference_model": True, "batch": batch}
    response = requests.post(url, headers=headers, json=data)
    return response


def update_model(
    model, url=f"http://127.0.0.1:{arbor_port}/v1/fine_tuning/grpo/update_model"
):
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers)
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

    dataset = load_dataset("trl-lib/tldr", split="train")
    current_model = "Qwen/Qwen3-0.6B"
    initialize_response = initialize_grpo(model=current_model)
    last_checkpoint = None

    for i in range(len(dataset)):
        inputs = dataset[i]
        input_messages = [{"role": "user", "content": inputs["prompt"]}]
        response = client.chat.completions.create(
            model=current_model, messages=input_messages, temperature=0.7, n=8
        )
        completions = [
            {"content": choice.message.content, "role": choice.message.role}
            for choice in response.choices
        ]
        rewards = _reward_func(inputs["prompt"], [c["content"] for c in completions])
        print(rewards)

        batch = []
        for completion, reward in zip(completions, rewards):
            batch.append(
                {"messages": input_messages, "completion": completion, "reward": reward}
            )
        step_response = run_grpo_step(model_name=current_model, batch=batch)
        current_model = step_response.json()["current_model"]

        if i % 10 == 0:
            update_response = update_model()
            current_model = update_response.json()["current_model"]

            checkpoint_response = checkpoint(checkpoint_name=f"checkpoint_{i}")
            last_checkpoint_name = checkpoint_response.json()["last_checkpoint_name"]
            import pdb

            pdb.set_trace()

    terminate_response = terminate_grpo()
    import pdb

    pdb.set_trace()


if __name__ == "__main__":
    main()
