import os
import sys
from typing import Dict, List

import pytest

from arbor.server.services.managers.gpu_manager import NoGPUsDetectedError

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture(autouse=True)
def mock_gpu_detection(monkeypatch) -> Dict[str, List[int]]:
    """Mock GPU detection to provide deterministic GPU state for tests."""

    state: Dict[str, List[int]] = {
        "total": [0, 1, 2],
        "free": [0, 1, 2],
        "busy": [],
    }

    def _get_state() -> Dict[str, List[int]]:
        total = list(state["total"])
        if not total:
            raise NoGPUsDetectedError("No GPUs detected in test fixture")

        return {
            "total": total,
            "free": list(state["free"]),
            "busy": list(state["busy"]),
        }

    monkeypatch.setattr(
        "arbor.server.services.managers.gpu_manager.get_gpu_state",
        _get_state,
    )
    try:
        yield state
    finally:
        state["total"] = [0, 1, 2]
        state["free"] = [0, 1, 2]
        state["busy"] = []
