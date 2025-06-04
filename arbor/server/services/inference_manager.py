import asyncio
import os
import random
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

import psutil
import requests

from arbor.server.core.config import Settings
from arbor.server.services.inference.vllm_client import VLLMClient


class InferenceManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.process = None
        self.launch_kwargs = {}
        self.last_activity = None
        self._shutting_down = False
        self.launched_model = None
        self.inference_count = 0
        self._session = None
        self.port = None
        self.group_port = None
        self.vllm_client = None
        self._is_updating = 0  # Counter for weight updates in progress
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        if self._shutting_down:
            print("\nForced exit during cleanup...")
            os._exit(1)

        print("\nReceived signal to terminate. Cleaning up...")
        self._shutting_down = True
        self.kill()
        sys.exit(0)

    def is_server_running(self):
        return self.process is not None

    def launch(self, model: str, launch_kwargs: Optional[Dict[str, Any]] = None):
        if self.is_server_running():
            print("Server is already launched.")
            return

        launch_kwargs = launch_kwargs or self.launch_kwargs

        prefixes = ["openai/", "huggingface/", "local:", "arbor:"]
        for prefix in prefixes:
            if model.startswith(prefix):
                model = model[len(prefix) :]

        print(f"Grabbing a free port to launch a vLLM server for model {model}")
        self.port = get_free_port()
        timeout = launch_kwargs.get("timeout", 1800)
        my_env = os.environ.copy()
        my_env["CUDA_VISIBLE_DEVICES"] = self.settings.arbor_config.inference.gpu_ids
        n_gpus = self.settings.arbor_config.inference.gpu_ids.count(",") + 1
        command = f"python -m arbor.server.services.inference.vllm_serve --model {model} --port {self.port} --gpu-memory-utilization 0.9 --tensor-parallel-size {n_gpus} --enable_prefix_caching True"

        if launch_kwargs.get("max_context_length"):
            command += f" --max_model_len {launch_kwargs['max_context_length']}"

        print(f"Running command: {command}")

        # We will manually stream & capture logs.
        process = subprocess.Popen(
            command.replace("\\\n", " ").replace("\\", " ").split(),
            text=True,
            stdout=subprocess.PIPE,  # We'll read from pipe
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            env=my_env,
        )

        # A threading.Event to control printing after the server is ready.
        # This will store *all* lines (both before and after readiness).
        print(f"vLLM server process started with PID {process.pid}.")
        stop_printing_event = threading.Event()
        logs_buffer = []

        def _tail_process(proc, buffer, stop_event):
            while True:
                line = proc.stdout.readline()
                if not line and proc.poll() is not None:
                    # Process ended and no new line
                    break
                if line:
                    buffer.append(line)
                    # Print only if stop_event is not set
                    if not stop_event.is_set():
                        print(f"[vLLM LOG] {line}", end="")

        # Start a background thread to read from the process continuously
        thread = threading.Thread(
            target=_tail_process,
            args=(process, logs_buffer, stop_printing_event),
            daemon=True,
        )
        thread.start()

        # A convenience getter so the caller can see all logs so far (and future).
        def get_logs() -> str:
            # Join them all into a single string, or you might return a list
            return "".join(logs_buffer)

        # Let the user know server is up
        print(f"Server ready on random port {self.port}!")

        # self.launch_kwargs["api_base"] = f"http://localhost:{port}/v1"
        # self.launch_kwargs["api_key"] = "local"
        self.get_logs = get_logs
        self.process = process
        self.thread = thread
        self.launched_model = model

        # Get another free port for weight sync group communication
        self.group_port = get_free_port()
        self.vllm_client = VLLMClient(
            port=self.port,
            group_port=self.group_port,
            connection_timeout=300,  # 5 minutes
        )

        # Once server is ready, we tell the thread to stop printing further lines.
        stop_printing_event.set()

    def kill(self):
        if self.process is None:
            print("No running server to kill.")
            return

        process = self.process
        thread = self.thread

        # Clear references first
        self.process = None
        self.thread = None
        self.get_logs = None
        self.last_activity = None

        try:
            kill_vllm_server(process.pid)
        except Exception as e:
            print(f"Error during cleanup: {e}")
            try:
                process.kill()  # Final attempt to kill
            except:
                pass

        print("Server killed.")

    async def run_inference(self, request_json: dict):
        # Check if weights are being updated
        while self.is_updating:
            # weights are being updated...waiting
            # print("Weights are being updated, waiting...")
            await asyncio.sleep(1)  # Small sleep to prevent busy waiting

        model = request_json["model"]
        prefixes = ["openai/", "huggingface/", "local:", "arbor:"]
        for prefix in prefixes:
            if model.startswith(prefix):
                model = model[len(prefix) :]
        print(f"Running inference for model {model}")

        # Monkeypatch for GRPO runs:
        # vllm complains if we don't give it the exact model name that was launched
        # TODO: This should really throw an error unless in a GRPO run.
        if model != self.launched_model:
            # print(f"Model changed from {model} to {self.current_model}")
            model = self.launched_model
            request_json["model"] = model

        # Update last_activity timestamp
        self.last_activity = datetime.now()

        if self.process is None:
            raise RuntimeError("Server is not running. Please launch it first.")

        return await self.vllm_client.chat(json_body=request_json)

    def start_weight_update(self):
        """Block inference during weight updates"""
        self._is_updating += 1

    def complete_weight_update(self):
        """Allow inference after weight update is complete"""
        self._is_updating = max(0, self._is_updating - 1)  # Prevent going negative

    @property
    def is_updating(self):
        """Check if any weight updates are in progress"""
        return self._is_updating > 0


def get_free_port() -> int:
    """
    Return a randomly selected free TCP port on localhost from a selection of 3-4 ports.
    """
    import random
    import socket

    ports = []
    for _ in range(random.randint(5, 10)):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", 0))
                ports.append(s.getsockname()[1])
        except Exception as e:
            print(f"Error binding to port: {e}")
    return random.choice(ports)


def wait_for_server(base_url: str, timeout: int = None) -> None:
    """
    Wait for the server to be ready by polling the /v1/models endpoint.

    Args:
        base_url: The base URL of the server (e.g. http://localhost:1234)
        timeout: Maximum time to wait in seconds. None means wait forever.
    """
    start_time = time.time()
    while True:
        try:
            response = requests.get(
                f"{base_url}/v1/models",
                headers={"Authorization": "Bearer None"},
            )
            if response.status_code == 200:
                # A small extra sleep to ensure server is fully up.
                time.sleep(5)
                break

            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError("Server did not become ready within timeout period")
        except requests.exceptions.RequestException:
            # Server not up yet, wait and retry
            time.sleep(1)


def kill_vllm_server(main_process_pid):
    try:
        # Get the parent process
        parent = psutil.Process(main_process_pid)

        # Get all child processes recursively
        children = parent.children(recursive=True)

        # Send SIGTERM to all child processes first
        for child in children:
            child.send_signal(signal.SIGTERM)

        # Send SIGTERM to parent process
        parent.send_signal(signal.SIGTERM)

        # Wait for processes to terminate gracefully
        gone, alive = psutil.wait_procs(children + [parent], timeout=10)

        # If any processes are still alive, force kill them
        for p in alive:
            p.kill()  # SIGKILL

    except psutil.NoSuchProcess:
        print(f"Process {main_process_pid} not found")
    except Exception as e:
        print(f"Error killing processes: {e}")
