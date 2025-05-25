from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

dataset = load_dataset("trl-lib/tldr", split="train")

# Convert the dataset to chat format
def convert_to_chat_format(example):
    # Extract the post content and TL;DR
    post = example['prompt']
    tldr = example['completion']
    
    # Create chat messages
    messages = [
        {"role": "user", "content": post},
        {"role": "assistant", "content": tldr}
    ]
    
    return {
        "prompt": messages,
        "completion": [{"role": "assistant", "content": tldr}]
    }

# Transform the dataset
chat_dataset = dataset.map(convert_to_chat_format)

# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion["content"])) for completion in completions]

training_args = GRPOConfig(
    output_dir="Qwen2-0.5B-GRPO", 
    logging_steps=10,
    max_prompt_length=2048,  # Adjust these based on your needs
    max_completion_length=512
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=chat_dataset,
)

trainer.train()