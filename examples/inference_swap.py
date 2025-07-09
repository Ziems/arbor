from openai import OpenAI

arbor_port = 7453

client = OpenAI(
    base_url=f"http://127.0.0.1:{arbor_port}/v1",  # Using Arbor server
    api_key="not-needed",  # If you're using a local server, you dont need an API key
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    temperature=0.7,
)
print(response)

# response = client.chat.completions.create(
#     model="Qwen/Qwen3-1.7B",
#     messages=[{"role": "user", "content": "Hello, how are you?"}],
#     temperature=0.7,
# )
# print(response)
