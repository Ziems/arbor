# train_grpo.py
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import queue
from torch.utils.data import Dataset
import time
import threading
from functools import lru_cache
from accelerate.utils import broadcast_object_list, gather, gather_object
import json
import os
from pathlib import Path

tldr_dataset = load_dataset("trl-lib/tldr", split="train")

# Dummy writer process to simulate real-time data
def dummy_writer(output_file="data_stream.jsonl"):
    print(f"Starting dummy writer, writing to {output_file}")
    for i, item in enumerate(tldr_dataset):
        with open(output_file, 'a') as f:
            f.write(json.dumps(item) + '\n')
        print(f"Wrote item {i} to {output_file}")
        time.sleep(5)  # Write a new item every 5 seconds

class JSONLStreamReader:
    def __init__(self, jsonl_file):
        self.jsonl_file = jsonl_file
        self.last_position = 0

    def read_new_entries(self):
        """Read any new entries from the file"""
        if not os.path.exists(self.jsonl_file):
            return []

        new_entries = []
        with open(self.jsonl_file, 'r') as f:
            f.seek(self.last_position)
            new_lines = f.readlines()

            if new_lines:
                self.last_position = f.tell()
                for line in new_lines:
                    try:
                        new_entries.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping invalid JSON line: {line.strip()}")

        return new_entries

class DataProducer:
    def __init__(self, jsonl_file="data_stream.jsonl"):
        self.reader = JSONLStreamReader(jsonl_file)
        self.queue = queue.Queue()
        self.running = False
        self.thread = None

    def start(self):
        """Start the producer thread"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._produce, daemon=True)
            self.thread.start()

    def stop(self):
        """Stop the producer thread"""
        self.running = False
        if self.thread:
            self.thread.join()

    def _produce(self):
        """Main production loop"""
        while self.running:
            new_entries = self.reader.read_new_entries()
            for entry in new_entries:
                self.queue.put(entry)
            time.sleep(1)  # Configurable delay between checks

class BlockingDynamicDataset(Dataset):
    def __init__(self, accelerator, size=1000, maxsize=100, jsonl_file="data_stream.jsonl"):
        self.size = size
        self.accelerator = accelerator
        self.get_cached_data = lru_cache(maxsize=maxsize)(self._get_data)

        # Create and start the producer in the main process only
        if accelerator is not None and accelerator.is_main_process:
            self.producer = DataProducer(jsonl_file)
            self.producer.start()

    def __len__(self):
        return self.size

    def _get_data(self, idx):
        print(f"Item not cached, getting new data")
        if self.accelerator.is_main_process:
            print(f"Main process {self.accelerator.process_index} getting new data")
            new_data = self.producer.queue.get(block=True)
            return broadcast_object_list([new_data])[0]
        else:
            print(f"Other process {self.accelerator.process_index} waiting for data")
            return broadcast_object_list([None])[0]

    def __getitem__(self, idx):
        print(f"Process {self.accelerator.process_index} getting item {idx}")
        return self.get_cached_data(idx)

    def __del__(self):
        """Cleanup when the dataset is destroyed"""
        if hasattr(self, 'producer'):
            self.producer.stop()

# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]

training_args = GRPOConfig(output_dir="Qwen2-0.5B-GRPO", logging_steps=10)
trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=BlockingDynamicDataset(None),
)

# Start the dummy writer only in the main process
if trainer.accelerator.is_main_process:
    dummy_writer_thread = threading.Thread(
        target=dummy_writer,
        args=("data_stream.jsonl",),
        daemon=True
    )
    dummy_writer_thread.start()
    # Give the writer a moment to start creating data
    time.sleep(2)

trainer.train_dataset = BlockingDynamicDataset(
    trainer.accelerator,
    jsonl_file="data_stream.jsonl"
)
trainer.train()