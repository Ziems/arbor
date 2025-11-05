import time

import requests
from datasets import load_dataset
from openai import OpenAI

import arbor

ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")["train"]

arbor_server_info = arbor.init()
base_url = arbor_server_info["base_url"]
chat_base = f"{base_url}chat"
client = OpenAI(
    base_url=base_url,  # Using Arbor server
    api_key="not-needed",  # If you're using a local server, you dont need an API key
)


def launch_model(
    model_name: str, *, num_gpus: int = 1, max_seq_len: int | None = None
) -> str:
    """Launch the inference server for a model via Arbor's /launch endpoint."""

    payload: dict[str, object] = {"model": model_name, "num_gpus": num_gpus}
    if max_seq_len is not None:
        payload["max_seq_len"] = max_seq_len

    response = requests.post(f"{chat_base}/launch", json=payload)
    response.raise_for_status()
    launch_info = response.json()
    job_id = str(launch_info.get("job_id"))
    print(
        "Launched inference job",
        {
            "job_id": job_id,
            "model": launch_info.get("model"),
        },
    )
    return job_id


def kill_model(job_id: str) -> None:
    """Terminate a running inference job via Arbor's /kill endpoint."""

    response = requests.post(
        f"{chat_base}/kill",
        json={"job_id": job_id},
    )
    response.raise_for_status()
    print("Terminated inference job", job_id)


def send_request(i: int, model_name: str) -> None:
    question = ds[i]["question"]
    # We don't need the response, just send the request
    client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": question}],
        temperature=0.7,
    )


def main() -> None:
    model_a = "Qwen/Qwen2.5-7B-Instruct"
    model_b = "Qwen/Qwen3-1.7B"

    # Launch first model and send a warm-up request
    job_a = launch_model(model_a, num_gpus=2)
    send_request(0, model_a)

    # Launch a second model and test it
    job_b = launch_model(model_b, num_gpus=1)
    send_request(1, model_b)

    # Tear down the first model and ensure the second still works
    kill_model(job_a)
    tik = time.time()
    send_request(2, model_b)
    tok = time.time()
    print(f"Time taken after model swap: {tok - tik} seconds")

    # Clean up the second model
    kill_model(job_b)


if __name__ == "__main__":
    try:
        main()
    finally:
        arbor.shutdown()
