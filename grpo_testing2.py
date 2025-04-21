# train_grpo.py
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import queue
from torch.utils.data import Dataset
import time
import threading
from functools import lru_cache
from accelerate.utils import broadcast_object_list, gather, gather_object

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
    def __init__(self, accelerator, size=1000, maxsize=100):
        self.size = size
        self.accelerator = accelerator
        self.get_cached_data = lru_cache(maxsize=maxsize)(self._get_data)

        # Start producer in a background thread (main process only)
        if accelerator is not None and accelerator.is_main_process:
            threading.Thread(target=data_producer, daemon=True).start()

    def __len__(self):
        return self.size

    def _get_data(self, idx):
        print(f"Item not cached, getting new data")
        # Main process blocks until new data arrives
        if self.accelerator.is_main_process:
            print(f"Main process {self.accelerator.process_index} getting new data")
            # Block until data is available
            new_data = data_queue.get(block=True)
            # Broadcast the data to all processes using broadcast_object_list
            return broadcast_object_list([new_data])[0]
        else:
            print(f"Other process {self.accelerator.process_index} waiting for data")
            # Other processes wait to receive the broadcast data
            return broadcast_object_list([None])[0]

    def __getitem__(self, idx):
        print(f"Process {self.accelerator.process_index} getting item {idx}")
        return self.get_cached_data(idx)

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