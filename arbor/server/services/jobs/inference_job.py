import asyncio
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional

import psutil
import requests

from arbor.server.core.config import Config
from arbor.server.services.inference.vllm_client import VLLMClient
from arbor.server.services.jobs.inference_launch_config import InferenceLaunchConfig
from arbor.server.services.jobs.job import Job
from arbor.server.utils.helpers import get_free_port
from arbor.server.utils.logging import get_logger
from arbor.server.utils.mock_utils import get_vllm_serve_module, setup_mock_environment

logger = get_logger(__name__)


class InferenceJob(Job):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.launch_config: InferenceLaunchConfig = None
        self.last_activity = None
        self._shutting_down = False
        self.inference_count = 0
        self._session = None
        self.port: Optional[int] = None
        self.group_port = None
        self.vllm_client = None
        self._is_updating = 0  # Counter for weight updates in progress
        self.launched_model_name = (
            None  # the name of the model that was originally launched
        )

    def is_server_running(self) -> bool:
        """Check if vLLM server is running."""
        return self.process is not None and self.process.poll() is None

    def launch(self, model: str, launch_config: InferenceLaunchConfig):
        self.launched_model_name = model
        if self.is_server_running():
            logger.info("Server is already launched.")
            return

        launch_config = launch_config or self.launch_config

        logger.info(f"Grabbing a free port to launch a vLLM server for model {model}")
        self.port = get_free_port()
        my_env = os.environ.copy()

        # Convert gpu_ids list to comma-separated string for environment variable
        gpu_ids_str = ",".join(map(str, launch_config.gpu_ids))
        my_env["CUDA_VISIBLE_DEVICES"] = gpu_ids_str
        
        # Setup mock environment if needed
        my_env = setup_mock_environment(my_env)
        
        n_gpus = len(launch_config.gpu_ids)
        vllm_module = get_vllm_serve_module()
        command = f"{sys.executable} -m {vllm_module} --model {self.launched_model_name} --port {self.port} --gpu-memory-utilization 0.9 --data-parallel-size {n_gpus} --enable_prefix_caching"

        if launch_config.max_context_length:
            command += f" --max_model_len {launch_config.max_context_length}"

        logger.info(f"Running command: {command}")

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
        logger.info(f"vLLM server process started with PID {process.pid}.")
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
                    # Log only if stop_event is not set
                    if not stop_event.is_set():
                        logger.info(f"[vLLM LOG] {line.strip()}")
                    else:
                        logger.debug(f"[vLLM LOG] {line.strip()}")

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

        self.get_logs = get_logs
        self.process = process
        self.thread = thread

        # Wait for the OpenAI API endpoints to be ready
        logger.info("Waiting for vLLM OpenAI API to be ready...")
        wait_for_server(f"http://localhost:{self.port}", timeout=300)
        logger.info(f"vLLM server ready on port {self.port}!")

        # Get another free port for weight sync group communication
        self.group_port = get_free_port()
        self.vllm_client = VLLMClient(
            port=self.port,
            group_port=self.group_port,
            connection_timeout=0,  # No additional timeout since we already waited
        )

        # Once server is ready, we tell the thread to stop printing further lines.
        stop_printing_event.set()

    def kill(self):
        if self.process is None:
            logger.info("No running server to kill.")
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
            logger.error(f"Error during cleanup: {e}")
            try:
                process.kill()  # Final attempt to kill
            except:
                pass

        logger.info("Server killed.")

    async def run_inference(self, request_json: dict):
        requested_model = request_json["model"]
        request_json["model"] = self.launched_model_name
        # Check if weights are being updated
        while self._is_updating:
            # weights are being updated...waiting
            await asyncio.sleep(1)  # Small sleep to prevent busy waiting

        # Update last_activity timestamp
        self.last_activity = datetime.now()

        if self.process is None:
            raise RuntimeError("Server is not running. Please launch it first.")
        response = await self.vllm_client.chat(json_body=request_json)
        response["model"] = (
            requested_model  # Set the model back to the original requested model
        )
        return response

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
        logger.warning(f"Process {main_process_pid} not found")
    except Exception as e:
        logger.error(f"Error killing processes: {e}")
