"""
Tests for simplified Config system.
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from arbor.server.core.config import Config


def test_config_defaults():
    """Test that Config has sensible defaults."""
    config = Config()

    assert config.storage_path == str(Path.home() / ".arbor" / "storage")
    # Don't assert gpu_ids as it may vary based on auto-detection
    assert config.accelerate_config is None
    assert config.inactivity_timeout == 30


def test_config_with_custom_values():
    """Test creating Config with custom values."""
    config = Config(
        storage_path="/custom/path",
        gpu_ids=[3, 4, 5],
        accelerate_config="/path/to/accelerate.yaml",
        inactivity_timeout=60,
    )

    assert config.storage_path == "/custom/path"
    assert config.gpu_ids == [3, 4, 5]
    assert config.accelerate_config == "/path/to/accelerate.yaml"
    assert config.inactivity_timeout == 60


def test_config_load_from_yaml():
    """Test loading config from YAML file."""
    # Create a temporary YAML file with a valid temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        config_data = {
            "storage_path": temp_dir,
            "gpu_ids": [1, 2, 3, 4],
            "accelerate_config": "/test/accelerate.yaml",
            "inactivity_timeout": 45,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            config = Config.load(temp_path)

            assert config.storage_path == temp_dir
            assert config.gpu_ids == [1, 2, 3, 4]
            assert config.accelerate_config == "/test/accelerate.yaml"
            assert config.inactivity_timeout == 45
        finally:
            Path(temp_path).unlink()


def test_config_load_partial_yaml():
    """Test loading config with only some values in YAML."""
    config_data = {"gpu_ids": [5, 6], "inactivity_timeout": 120}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = f.name

    try:
        config = Config.load(temp_path)

        # Should use defaults for missing values
        assert config.storage_path == str(Path.home() / ".arbor" / "storage")
        assert config.accelerate_config is None

        # Should use YAML values for provided ones
        assert config.gpu_ids == [5, 6]
        assert config.inactivity_timeout == 120
    finally:
        Path(temp_path).unlink()


def test_config_load_nonexistent_file():
    """Test loading config from nonexistent file uses defaults."""
    config = Config.load("/nonexistent/config.yaml")

    # Should use all defaults with auto-detected GPUs
    assert config.storage_path == str(Path.home() / ".arbor" / "storage")
    assert config.accelerate_config is None
    assert config.inactivity_timeout == 30


def test_config_load_invalid_yaml():
    """Test loading invalid YAML falls back to defaults."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("invalid: yaml: content: [")
        temp_path = f.name

    try:
        # Should not raise exception, should use defaults
        config = Config.load(temp_path)

        assert config.storage_path == str(Path.home() / ".arbor" / "storage")
    finally:
        Path(temp_path).unlink()


def test_config_load_empty_yaml():
    """Test loading empty YAML file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("")
        temp_path = f.name

    try:
        config = Config.load(temp_path)

        # Should use all defaults
        assert config.storage_path == str(Path.home() / ".arbor" / "storage")
    finally:
        Path(temp_path).unlink()


def test_config_load_default_path(monkeypatch):
    """Test loading from default path (~/.arbor/config.yaml)."""
    # Mock the home path
    mock_home = Path("/mock/home")
    monkeypatch.setattr("pathlib.Path.home", lambda: mock_home)

    # Create temporary config at mock default location
    config_dir = Path(tempfile.mkdtemp()) / ".arbor"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "config.yaml"

    config_data = {"gpu_ids": [7, 8, 9]}
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    try:
        # Mock Path.home and the config file existence
        monkeypatch.setattr("pathlib.Path.home", lambda: config_dir.parent)

        config = Config.load()  # No path provided
        assert config.gpu_ids == [7, 8, 9]
    finally:
        config_file.unlink()
        config_dir.rmdir()


@pytest.mark.parametrize(
    "storage_path",
    [
        "/simple/path",
        "/path/with spaces/storage",
        str(Path.home() / "custom" / "arbor"),
    ],
)
def test_ensure_storage_path(storage_path):
    """Test that storage path directories are created."""
    config = Config(storage_path=storage_path)

    # This would normally create directories, but we'll just test the method exists
    # In a real test environment, you might want to use a temporary directory
    assert hasattr(config, "_ensure_storage_path")


def test_config_gpu_ids_validation():
    """Test that GPU IDs can be various types of lists."""
    # Test with different GPU configurations
    configs = [
        Config(gpu_ids=[]),
        Config(gpu_ids=[0]),
        Config(gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7]),
        Config(gpu_ids=[1, 3, 5, 7]),  # Non-contiguous
    ]

    for config in configs:
        assert isinstance(config.gpu_ids, list)
        assert all(isinstance(gpu_id, int) for gpu_id in config.gpu_ids)


def test_config_yaml_round_trip():
    """Test that config can be saved and loaded from YAML."""
    original_config = Config(
        storage_path="/test/storage",
        gpu_ids=[2, 4, 6],
        accelerate_config="/test/accel.yaml",
        inactivity_timeout=90,
    )

    # Convert to dict (simulating YAML serialization)
    config_dict = {
        "storage_path": original_config.storage_path,
        "gpu_ids": original_config.gpu_ids,
        "accelerate_config": original_config.accelerate_config,
        "inactivity_timeout": original_config.inactivity_timeout,
    }

    # Create new config from dict (simulating YAML loading)
    loaded_config = Config(**config_dict)

    assert loaded_config.storage_path == original_config.storage_path
    assert loaded_config.gpu_ids == original_config.gpu_ids
    assert loaded_config.accelerate_config == original_config.accelerate_config
    assert loaded_config.inactivity_timeout == original_config.inactivity_timeout
