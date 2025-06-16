import time
from typing import Optional


class IngestionMonitor:
    """Monitor for tracking timing of training steps and data ingestion (queue pops)"""

    def __init__(self):
        self._last_step_time: Optional[float] = None
        self._last_queue_pop_time: Optional[float] = None

    def time_since_last_step(self) -> float:
        """Get time elapsed since last training step"""
        if self._last_step_time is None:
            return float("inf")
        return time.time() - self._last_step_time

    def time_since_last_queue_pop(self) -> float:
        """Get time elapsed since last queue pop"""
        if self._last_queue_pop_time is None:
            return float("inf")
        return time.time() - self._last_queue_pop_time

    def set_last_queue_pop_time(self, timestamp: Optional[float] = None) -> None:
        """Set the last queue pop time"""
        self._last_queue_pop_time = timestamp if timestamp is not None else time.time()

    def set_last_step_time(self, timestamp: Optional[float] = None) -> None:
        """Set the last step time"""
        self._last_step_time = timestamp if timestamp is not None else time.time()

    def reset(self) -> None:
        """Reset all timing data"""
        self._last_step_time = None
        self._last_queue_pop_time = None
