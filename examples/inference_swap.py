import time

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")["train"]

arbor_port = 7453

client = OpenAI(
    base_url=f"http://127.0.0.1:{arbor_port}/v1",  # Using Arbor server
    api_key="not-needed",  # If you're using a local server, you dont need an API key
)


from concurrent.futures import ThreadPoolExecutor


def send_request(i):
    question = ds[i]["question"]
    # We don't need the response, just send the request
    client.chat.completions.create(
        model="Qwen/Qwen2.5-7B",  # Changed from 0.5B to 7B
        messages=[{"role": "user", "content": question}],
        temperature=0.7,
    )


send_request(0)  # First one takes forever

tik = time.time()
with ThreadPoolExecutor(max_workers=100) as executor:  # Essentially unlimited workers
    list(tqdm(executor.map(send_request, range(1000)), total=1000))
tok = time.time()
print(f"Time taken: {tok - tik} seconds")

# response = client.chat.completions.create(
#     model="Qwen/Qwen3-1.7B",
#     messages=[{"role": "user", "content": "Hello, how are you?"}],
#     temperature=0.7,
# )
# print(response)
