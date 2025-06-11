import logging
import time
from functools import lru_cache
from typing import Any, Dict, List

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
    ):
        self._buffer: List[Dict[str, Any]] = []
        self._logger = logging.getLogger(__name__)

    def set_accelerator(self, accelerator: Accelerator):
        self.accelerator = accelerator

    def set_comms_handler(self, comms_handler: ArborScriptCommsHandler):
        self.comms_handler = comms_handler

    def __len__(self) -> int:
        return 1_000_000

    def _fill_buffer(self, target_size: int) -> None:
        while len(self._buffer) < target_size:
            try:
                if self.comms_handler is None:
                    raise ValueError("comms_handler is not initialized")

                group = self.comms_handler.receive_data()
                if group is not None:
                    self._logger.debug("Received group from comms handler")
                    [self._buffer.append(item) for sublist in group for item in sublist]

            except Exception as e:
                if "Context was terminated" in str(e):
                    self._logger.error(
                        "ZMQ context was terminated while filling buffer"
                    )
                    raise RuntimeError("ZMQ context was terminated") from e
                self._logger.warning(f"Error receiving data: {e}")
                continue

    def _transform_batch(self, items: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        if not items:
            raise ValueError("Cannot transform empty batch")

        return {key: [item[key] for item in items] for key in items[0].keys()}

    def __getitem__(self, idx: List[int]) -> Dict[str, List[Any]]:
        if self.accelerator is None:
            self._logger.error("Accelerator not initialized")
            raise ValueError("Accelerator must be initialized before getting items")
        if self.comms_handler is None:
            self._logger.error("Comms handler not initialized")
            raise ValueError("Comms handler must be initialized before getting items")

        batch_size = len(idx)
        if batch_size == 0:
            raise ValueError("Batch size must be greater than 0")

        try:
            self._fill_buffer(batch_size)

            if len(self._buffer) < batch_size:
                raise RuntimeError(
                    f"Not enough items in buffer (got {len(self._buffer)}, need {batch_size})"
                )

            batch_items = self._buffer[:batch_size]
            self._buffer = self._buffer[batch_size:]

            return self._transform_batch(batch_items)

        except Exception as e:
            self._logger.error(f"Error getting batch: {e}")
            raise
