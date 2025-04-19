# train_grpo.py
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import queue
from torch.utils.data import Dataset
import time
import threading

tldr_dataset = load_dataset("trl-lib/tldr", split="train")

data_queue = queue.Queue()

# Data producer (simulates real-time data arrival)
def data_producer():
    # Get individual examples from tldr_dataset
    for i in range(len(tldr_dataset)):  # Process one sample at a time
        sample = tldr_dataset[i]
        # Put individual sample in the queue
        data_queue.put(sample)
        time.sleep(1)  # Keep the 5 second delay between samples



# Custom Dataset that blocks until new data arrives
class BlockingDynamicDataset(Dataset):
    def __init__(self, accelerator, size=1000):
        self.size = size
        self.current_data = None  # Will be updated with new data
        self.accelerator = accelerator

        # Start producer in a background thread (main process only)
        if accelerator is not None and accelerator.is_main_process:
            threading.Thread(target=data_producer, daemon=True).start()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Main process blocks until new data arrives
        # Print accelerator process ID
        print(f"Accelerator process ID: {self.accelerator.process_index}")
        if self.accelerator.is_main_process:
            # Block until data is available (no timeout for true blocking)
            import pdb; pdb.set_trace()
            self.current_data = data_queue.get(block=True)
            # Reset index for new data
        # Broadcast data to all processes

        # Convert to tensor format expected by Trainer
        return self.current_data

dataset = BlockingDynamicDataset(None)

# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]

training_args = GRPOConfig(output_dir="Qwen2-0.5B-GRPO", logging_steps=10)
trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
)
trainer.train_dataset = BlockingDynamicDataset(trainer.accelerator)
trainer.train()