import json
import logging
import time
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional

from accelerate import Accelerator
from datasets import Dataset as HuggingFaceDataset
from torch.utils.data import Dataset as TorchDataset

from arbor.server.services.comms.comms import ArborScriptCommsHandler

logger = logging.getLogger(__name__)


class BlockingRotatingQueueDataset(TorchDataset):
    def __init__(
        self,
        size=10_000,  # Just a random number
        maxsize=100,
        ingestion_monitor: Optional["IngestionMonitor"] = None,
    ):
        self.size = size
        # Use a regular cache dict instead of lru_cache to avoid unhashable type issues
        self._data_cache = {}
        self._cache_maxsize = maxsize
        self.completion_counters = {}
        self.ingestion_monitor = ingestion_monitor
        self.accelerator = None
        self.comms_handler = None

    def set_accelerator(self, accelerator: Accelerator):
        self.accelerator = accelerator

    def set_comms_handler(self, comms_handler: ArborScriptCommsHandler):
        self.comms_handler = comms_handler

    def __len__(self):
        return self.size

    def _get_data(self, idx):
        rank = self.accelerator.process_index
        world_size = self.accelerator.num_processes

        if self.accelerator.is_main_process and self.ingestion_monitor:
            self.ingestion_monitor.set_last_queue_pop_time()

        if idx not in self.completion_counters:
            self.completion_counters[idx] = 0

        try:
            new_data = self.comms_handler.receive_data()

        except Exception as e:
            print(f"[rank {rank}] Error receiving data: {e}")
            if "unhashable" in str(e):
                print(
                    f"[rank {rank}] DEBUGGING: Unhashable type error in data reception"
                )
                print(
                    f"[rank {rank}] This might be related to caching or data structure issues"
                )
            new_data = None

        return new_data

    def get_cached_data(self, idx):
        """Get data with simple dictionary caching instead of lru_cache"""
        if idx in self._data_cache:
            return self._data_cache[idx]

        # If cache is full, clear oldest entries (simple FIFO)
        if len(self._data_cache) >= self._cache_maxsize:
            # Remove first half of cache entries
            keys_to_remove = list(self._data_cache.keys())[: self._cache_maxsize // 2]
            for key in keys_to_remove:
                del self._data_cache[key]

        # Get new data and cache it
        data = self._get_data(idx)
        self._data_cache[idx] = data
        return data

    def __getitem__(self, idx):
        print(f"Getting item {idx}")
        data = self.get_cached_data(idx)

        if data is None:
            return None

        counter = self.completion_counters.get(idx, 0)
        item = data[counter]
        self.completion_counters[idx] = (counter + 1) % len(data)
        return item


class BlockingQueueDataset(HuggingFaceDataset):
    def __init__(
        self,
        ingestion_monitor: Optional["IngestionMonitor"] = None,
    ):
        self._buffer: List[Dict[str, Any]] = []
        self._logger = logging.getLogger(__name__)
        self.ingestion_monitor = ingestion_monitor

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
                    for trajectory in group:
                        trajectory_copy = json.loads(json.dumps(trajectory))
                        for item in trajectory:
                            item["trajectory"] = trajectory_copy
                            self._buffer.append(item)

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

            if self.ingestion_monitor:
                self.ingestion_monitor.set_last_queue_pop_time()

            return self._transform_batch(batch_items)

        except Exception as e:
            self._logger.error(f"Error getting batch: {e}")
            raise
