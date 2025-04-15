import threading
import subprocess
import socket
import time
import requests
import torch
from typing import Optional, Dict, Any
from datetime import datetime
from arbor.server.core.config import Settings


class InferenceManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.process = None
        self.launch_kwargs = {}
        self.last_activity = None

    def is_server_running(self):
        return self.process is not None

    def launch(self, model: str, launch_kwargs: Optional[Dict[str, Any]] = None):
        if self.is_server_running():
            print("Server is already launched.")
            return

        launch_kwargs = launch_kwargs or self.launch_kwargs

        if model.startswith("openai/"):
            model = model[7:]
        if model.startswith("local:"):
            model = model[6:]
        if model.startswith("huggingface/"):
            model = model[len("huggingface/"):]

        import os
        print(f"Grabbing a free port to launch an vLLM server for model {model}")
        print(
            f"We see that CUDA_VISIBLE_DEVICES is {os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}"
        )
        port = get_free_port()
        timeout = launch_kwargs.get("timeout", 1800)
        command = f"vllm serve {model} --port {port} --gpu-memory-utilization=0.6"
        print(f"Running command: {command}")

        # We will manually stream & capture logs.
        process = subprocess.Popen(
            command.replace("\\\n", " ").replace("\\", " ").split(),
            text=True,
            stdout=subprocess.PIPE,  # We'll read from pipe
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
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
                        print(line, end="")

        # Start a background thread to read from the process continuously
        thread = threading.Thread(
            target=_tail_process,
            args=(process, logs_buffer, stop_printing_event),
            daemon=True,
        )
        thread.start()

        # Wait until the server is ready (or times out)
        base_url = f"http://localhost:{port}"
        try:
            wait_for_server(base_url, timeout=timeout)
        except TimeoutError:
            # If the server doesn't come up, we might want to kill it:
            process.kill()
            raise


        # Once server is ready, we tell the thread to stop printing further lines.
        stop_printing_event.set()

        # A convenience getter so the caller can see all logs so far (and future).
        def get_logs() -> str:
            # Join them all into a single string, or you might return a list
            return "".join(logs_buffer)

        # Let the user know server is up
        print(
            f"Server ready on random port {port}!"
        )

        self.launch_kwargs["api_base"] = f"http://localhost:{port}/v1"
        self.launch_kwargs["api_key"] = "local"
        self.get_logs = get_logs
        self.process = process
        self.thread = thread


    def kill(self):
        if self.process is None:
            print("No running server to kill.")
            return

        # Store a reference to process and thread before clearing
        process = self.process
        thread = self.thread

        # Clear the references first
        self.process = None
        self.thread = None
        self.get_logs = None
        self.last_activity = None

        # Then terminate the process and join thread
        process.terminate()  # Send SIGTERM signal
        process.wait()  # Wait for process to finish
        thread.join()
        print("Server killed.")

    def run_inference(self, request_json: dict):
        # Update last_activity timestamp
        self.last_activity = datetime.now()

        if self.process is None or self.launch_kwargs.get('api_base') is None:
            raise RuntimeError("Server is not running. Please launch it first.")

        url = f"{self.launch_kwargs['api_base']}/chat/completions"
        response = requests.post(url, json=request_json)
        return response.json()

    def update_named_param(self, name: str, weights: torch.Tensor):
        """
        Updates a specific named parameter in the model and broadcasts it to other processes.

        Args:
            name (`str`):
                Name of the layer whose weights are being updated.
            weights (`torch.Tensor`):
                Tensor containing the updated weights.
        """
        dtype, shape = str(weights.dtype), tuple(weights.shape)
        url = f"{self.launch_kwargs['api_base']}/update_named_param/"
        import pdb; pdb.set_trace()
        response = requests.post(url, json={"name": name, "dtype": dtype, "shape": shape})
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        # Broadcast the weights to the other processes
        # self.pynccl_comm.broadcast(weights, src=self.rank, stream=torch.cuda.current_stream())
        # self.pynccl_comm.group.barrier()

    def reset_prefix_cache(self):
        """
        Resets the prefix cache for the model.
        """
        url = f"{self.launch_kwargs['api_base']}/reset_prefix_cache/"
        response = requests.post(url)
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    def update_model(self, model):
        for name, param in model.named_parameters():
            self.update_named_param(name, param.data)
        self.reset_prefix_cache()


def get_free_port() -> int:
    """
    Return a free TCP port on localhost.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]

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
