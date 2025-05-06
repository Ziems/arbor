import time
from pathlib import Path

import pytest
import requests

from arbor.cli import start_server, stop_server


@pytest.fixture(scope="module")
def test_storage(tmp_path_factory):
    """Create a temporary storage directory for tests"""
    return tmp_path_factory.mktemp("test_storage")


def upload_file(url, file_path):
    with open(file_path, "rb") as file:
        files = {"file": file}
        response = requests.post(url, files=files)
    return response


def test_server_lifecycle(test_storage):
    # Start the server on a test port with temporary storage
    server = start_server(port=8000, storage_path=str(test_storage))

    try:
        # Test that the server is running by making a request
        test_file_path = Path(__file__).parent / "data" / "training_data_sft.jsonl"
        with open(test_file_path, "rb") as file:
            files = {"file": file}
            response = requests.post("http://localhost:8000/v1/files", files=files)
        assert response.status_code == 200

    finally:
        # Clean up: stop the server
        stop_server(server)


def test_multiple_servers(test_storage):
    # Test that we can run multiple servers on different ports
    server1 = start_server(port=8124, storage_path=str(test_storage))
    server2 = start_server(port=8125, storage_path=str(test_storage))

    try:

        # Test both servers are running
        test_file_path = Path(__file__).parent / "data" / "training_data_sft.jsonl"
        resp1 = upload_file("http://localhost:8124/v1/files", test_file_path)
        resp2 = upload_file("http://localhost:8125/v1/files", test_file_path)

        assert resp1.status_code == 200
        assert resp2.status_code == 200

    finally:
        # Clean up
        stop_server(server1)
        stop_server(server2)
