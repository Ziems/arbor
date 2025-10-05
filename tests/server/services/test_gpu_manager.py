import pytest

from arbor.server.core.config import Config
from arbor.server.services.managers.gpu_manager import GPUAllocationError, GPUManager


def test_manager_uses_detected_gpus_when_config_empty(mock_gpu_detection):
    mock_gpu_detection["total"] = [0, 1]
    mock_gpu_detection["free"] = [0, 1]

    manager = GPUManager(Config())

    assert manager.get_status()["total_gpus"] == [0, 1]


def test_allocate_respects_live_free_set(mock_gpu_detection):
    mock_gpu_detection["total"] = [0, 1, 2]
    mock_gpu_detection["free"] = [0, 2]
    mock_gpu_detection["busy"] = [1]

    manager = GPUManager(Config())
    allocated = manager.allocate_gpus("job", 2)

    assert set(allocated) == {0, 2}


def test_error_when_not_enough_free_gpus(mock_gpu_detection):
    mock_gpu_detection["total"] = [0, 1]
    mock_gpu_detection["free"] = [0]
    mock_gpu_detection["busy"] = [1]

    manager = GPUManager(Config())

    with pytest.raises(GPUAllocationError) as exc:
        manager.allocate_gpus("job", 2)

    message = str(exc.value)
    assert "Not enough free GPUs" in message
    assert "Busy" in message


def test_manager_raises_when_all_gpus_busy(mock_gpu_detection):
    mock_gpu_detection["total"] = [0, 1]
    mock_gpu_detection["free"] = []
    mock_gpu_detection["busy"] = [0, 1]

    with pytest.raises(GPUAllocationError) as exc:
        GPUManager(Config())

    assert "All GPUs are currently in use" in str(exc.value)


def test_manager_raises_when_no_gpus_detected(mock_gpu_detection):
    mock_gpu_detection["total"] = []
    mock_gpu_detection["free"] = []
    mock_gpu_detection["busy"] = []

    with pytest.raises(GPUAllocationError) as exc:
        GPUManager(Config())

    assert "No GPUs detected" in str(exc.value)
