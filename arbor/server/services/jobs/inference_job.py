import os
import signal
import sys
import time
from datetime import datetime
from typing import Optional
import psutil

import requests

from arbor.server.core.config import Config
from arbor.server.services.jobs.inference_launch_config import InferenceLaunchConfig
from arbor.server.services.comms.control_server import TrainerControlServer
from arbor.server.services.jobs.job import Job
from arbor.server.utils.helpers import get_free_port
from arbor.server.utils.logging import get_logger
from arbor.server.utils.mock_utils import (
    get_vllm_serve_module,
    setup_mock_environment,
    should_use_mock_gpu,
)
from arbor.server.utils.process_runner import InferenceProcessRunner

# Conditionally import the appropriate vLLM client
if should_use_mock_gpu():
    from arbor.server.services.inference.vllm_client_mock import VLLMClient
else:
    from arbor.server.services.inference.vllm_client import VLLMClient

logger = get_logger(__name__)


class InferenceJob(Job):
    def __init__(self, config: Config):
        super().__init__(
            config, prefix="infer"
        )  # Default artifacts=[JobArtifact.LOGS] is perfect for inference jobs
        self.process_runner: Optional[InferenceProcessRunner] = None
        self.launch_config: InferenceLaunchConfig = None
        self.last_activity = None
        self._shutting_down = False
        self.inference_count = 0
        self._session = None
        self.port: Optional[int] = None
        self.group_port = None
        self.vllm_client = None
        self._is_updating = 0  # Counter for weight updates in progress
        self._active_requests = 0  # Counter for currently active inference requests
        self.launched_model_name = (
            None  # the name of the model that was originally launched
        )
        self._is_grpo_sub_job = False  # Flag to track if this is a GRPO sub-job

    def _setup_directories(self, artifacts):
        """Override to skip directory creation for GRPO sub-jobs."""
        if getattr(self, "_is_grpo_sub_job", False):
            # Skip directory creation for GRPO sub-jobs
            # The parent GRPO job will handle directory setup
            return
        # Call parent method for standalone inference jobs
        super()._setup_directories(artifacts)

    def is_server_running(self) -> bool:
        """Check if vLLM server is running."""
        return self.process_runner is not None and self.process_runner.is_running()

    def launch(
        self,
        model: str,
        launch_config: InferenceLaunchConfig,
        trainer_controller: TrainerControlServer = None,
    ):
        self.launched_model_name = model
        self.trainer_controller = trainer_controller
        if self.is_server_running():
            logger.info("Server is already launched.")
            return

        launch_config = launch_config or self.launch_config

        # If this is a GRPO inference job, inherit the parent job's ID and don't create separate directories
        if launch_config.is_grpo and launch_config.grpo_job_id:
            self.id = f"{launch_config.grpo_job_id}-inference"
            self._is_grpo_sub_job = True
            assert (
                trainer_controller is not None
            ), "Trainer controller is required for GRPO inference jobs"
            self.trainer_controller = trainer_controller
            # Don't create separate directories for GRPO inference jobs
            # The log file path will be set by the parent GRPO job

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
        command = f"{sys.executable} -m {vllm_module} --model {self.launched_model_name} --port {self.port} --gpu-memory-utilization 0.9 --tensor-parallel-size {n_gpus} --enable_prefix_caching"

        if launch_config.max_context_length:
            command += f" --max_model_len {launch_config.max_context_length}"

        logger.info(f"Running command: {command}")

        # Start the inference server using our clean process runner
        self.process_runner = InferenceProcessRunner(self.id)
        process = self.process_runner.start_inference_server(
            command,
            env=my_env,
            log_callback=self.create_filtered_log_callback("INFERENCE"),
        )

        logger.info(f"vLLM server process started with PID {process.pid}.")

        # Store process reference for compatibility
        self.process = process

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

        # Server is ready, ProcessRunner handles ongoing logging

    def cancel(self):
        """Cancel the inference job"""

        # Call parent cancel method to check status and set CANCELLED
        super().cancel()

        logger.info(f"Cancelling InferenceJob {self.id}")

        # Terminate the inference server (no model saving for inference jobs)
        self.terminate(save_model=False)

    def terminate(self, save_model: bool = True):
        """Terminate the inference server

        Args:
            save_model: Not applicable for inference jobs, ignored
        """
        if self.process_runner is None:
            logger.info("No running server to terminate.")
            return

        logger.info(f"Terminating InferenceJob {self.id}")

        # Use ProcessRunner for clean termination
        self.process_runner.terminate()

        # Ensure vLLM worker subtree is terminated as well
        try:
            if (
                getattr(self, "process", None) is not None
                and self.process.poll() is None
            ):
                kill_vllm_server(self.process.pid)
        except Exception as e:
            logger.warning(f"Force-kill fallback for vLLM processes failed: {e}")

        self.process_runner = None
        self.process = None
        self.last_activity = None
        logger.info("Server terminated.")

    async def run_inference(self, request_json: dict):
        requested_model = request_json["model"]
        request_json["model"] = self.launched_model_name

        try:
            # Update last_activity timestamp
            self.last_activity = datetime.now()

            if self.process is None:
                raise RuntimeError("Server is not running. Please launch it first.")
            response = await self.vllm_client.chat(json_body=request_json)
            response["model"] = (
                requested_model  # Set the model back to the original requested model
            )
            return response
        except Exception as e:
            logger.error(f"Error running inference: {e}")

    def create_filtered_log_callback(self, job_type: str = "INFERENCE"):
        """Create a log callback that filters out verbose inference logs from terminal output"""
        import json
        from datetime import datetime

        def filtered_callback(line: str):
            timestamp = datetime.now()
            timestamp_iso = timestamp.isoformat()

            # Filter out verbose logs from terminal output
            should_suppress = any(
                [
                    "it/s" in line,  # Progress bars
                    "Adding requests:" in line,
                    "Processed prompts:" in line,
                    line.strip().startswith("INFO:")
                    and (' - "' in line and 'HTTP/1.1" ' in line),  # All HTTP requests
                    "est. speed input:" in line,
                    "est. speed output:" in line,
                    "%" in line
                    and ("█" in line or "░" in line),  # Progress bar characters
                ]
            )

            # Only log non-verbose messages to terminal
            if not should_suppress:
                logger.info(f"[{job_type} LOG] {line}")

            # Always save to log file (if path is set)
            if self.log_file_path:
                try:
                    os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)

                    log_entry = {
                        "timestamp": timestamp_iso,
                        "level": "info",
                        "job_id": self.id,
                        "job_type": job_type,
                        "message": line,
                        "source": "training_log",
                    }

                    with open(self.log_file_path, "a", encoding="utf-8") as log_file:
                        log_file.write(json.dumps(log_entry) + "\n")
                        log_file.flush()
                except Exception as e:
                    logger.error(
                        f"Failed to write to log file {self.log_file_path}: {e}"
                    )

        return filtered_callback


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
