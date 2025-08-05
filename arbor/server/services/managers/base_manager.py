"""
Base manager class that enforces cleanup methods for all managers.
"""

import abc
from typing import Any

from arbor.server.utils.logging import get_logger


class BaseManager(abc.ABC):
    """
    Base class for all managers in the Arbor server.

    All managers must implement a cleanup method to ensure proper
    resource cleanup when the server shuts down.
    """

    def __init__(self, config: Any = None):
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self._cleanup_called = False

    @abc.abstractmethod
    def cleanup(self) -> None:
        """
        Clean up all resources managed by this manager.

        This method should:
        1. Terminate any running processes
        2. Close any open connections
        3. Clean up temporary files
        4. Release any other resources

        This method should be idempotent (safe to call multiple times).
        """
        pass

    def __del__(self):
        """Ensure cleanup is called when manager is garbage collected."""
        if not self._cleanup_called:
            try:
                self.cleanup()
            except Exception as e:
                # Use print since logger might not be available during cleanup
                print(f"Error during {self.__class__.__name__} cleanup in __del__: {e}")
