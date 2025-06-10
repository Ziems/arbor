import time
from functools import lru_cache

from accelerate import Accelerator
from datasets import Dataset

from arbor.server.services.comms.comms import ArborScriptCommsHandler


class BlockingRotatingQueueDataset(Dataset):
    def __init__(
        self,
        accelerator: Accelerator,
        comms_handler: ArborScriptCommsHandler,
        size=10_000,  # Just a random number
        maxsize=100,
    ):
        self.size = size
        self.accelerator = accelerator
        self.comms_handler = comms_handler
        self.get_cached_data = lru_cache(maxsize=maxsize)(self._get_data)
        self.completion_counters = {}

    def __len__(self):
        return self.size

    def _get_data(self, idx):
        rank = self.accelerator.process_index
        world_size = self.accelerator.num_processes

        if self.accelerator.is_main_process:
            global last_queue_pop_time
            last_queue_pop_time = time.time()

        if idx not in self.completion_counters:
            self.completion_counters[idx] = 0

        try:
            new_data = self.comms_handler.receive_data()

        except Exception as e:
            print(f"[rank {rank}] Error receiving data: {e}")
            new_data = None

        return new_data

    def __getitem__(self, idx):
        data = self.get_cached_data(idx)
        # Create hash of data to detect if processes are using the same idx for the same data
        data_hash = format(abs(hash(str(data))) % (16**8), "08x")

        if data is None:
            return None

        counter = self.completion_counters.get(idx, 0)
        item = data[counter]
        self.completion_counters[idx] = (counter + 1) % len(data)
        return item


class BlockingQueueDataset(Dataset):
    def __init__(
        self,
        accelerator: Accelerator,
        comms_handler: ArborScriptCommsHandler,
        size=10_000,  # Just a random number
    ):
        self.size = size
        self.accelerator = accelerator
        self.comms_handler = comms_handler

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        item = self.comms_handler.receive_data()
        print(f"Received item: {item}")
        return item
