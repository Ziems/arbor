import threading
import subprocess
import socket
import time
import requests
import torch
from typing import Optional, Dict, Any
from datetime import datetime

from arbor.server.core.config import Settings
from arbor.server.services.vllm_client import VLLMClient


class InferenceManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.process = None
        self.launch_kwargs = {}
        self.last_activity = None
        self.vllm_client = None

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

        command = f"python src/arbor/server/services/vllm_serve.py --model {model} --port {port} --max_model_len 8192  --gpu_memory_utilization 0.6 --enable_prefix_caching True"
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

        self.vllm_client = VLLMClient(host="localhost", server_port=port, connection_timeout=timeout)

        # Wait until the server is ready (or times out)
        # try:
        #     self.vllm_client.check_server(timeout=timeout)
        # except TimeoutError:
        #     # If the server doesn't come up, we might want to kill it:
        #     process.kill()
        #     raise


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

        if self.process is None or self.vllm_client is None:
            raise RuntimeError("Server is not running. Please launch it first.")

        # url = f"{self.launch_kwargs['api_base']}/chat/completions"
        # response = requests.post(url, json=request_json)
        # return response.json()
        messages = request_json["messages"]
        response = self.vllm_client.chat(messages)
        import pdb; pdb.set_trace()

        return response


    def update_named_param(self, name: str, weights: torch.Tensor):
        return self.vllm_client.update_named_param(name, weights)

    def reset_prefix_cache(self):
        return self.vllm_client.reset_prefix_cache()

    def update_model(self, model):
        return self.vllm_client.update_model_params(model)


def get_free_port() -> int:
    """
    Return a free TCP port on localhost.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]