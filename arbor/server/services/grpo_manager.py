import asyncio
import json
import os
import random
import signal
import socket
import string
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from arbor.server.api.models.schemas import GRPOConfigRequest, GRPORequest
from arbor.server.core.config import Settings
from arbor.server.services.comms.comms import ArborServerCommsHandler
from arbor.server.services.inference_manager import InferenceManager


class GRPOManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.training_process = None
        self.current_model = None
        self.train_kwargs = None
        self.server_comms_handler = None
        self.status_thread = None
        self.model_saved_and_reload_requested = False

        self.data_count = 0
        self.last_inference_update = 0
        # Set up signal handler
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle keyboard interrupt (SIGINT) gracefully."""
        print("\nReceived keyboard interrupt. Shutting down gracefully...")
        self.terminate(None)
        sys.exit(0)

    def make_output_dir(
        self, model_name: str, run_suffix: Optional[str] = None
    ) -> tuple[str, str]:
        """Create a unique output directory name for the training run."""
        model_name = model_name.split("/")[-1].lower()
        suffix = (
            run_suffix
            if run_suffix
            else "".join(random.choices(string.ascii_letters + string.digits, k=6))
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"grpo:{model_name}:{suffix}:{timestamp}"
        return name, str(Path(self.settings.STORAGE_PATH).resolve() / "models" / name)

    def find_training_args(self, request: GRPOConfigRequest) -> dict:
        """Process the config request and return training arguments."""
        name, output_dir = self.make_output_dir(request.model, request.suffix)

        # Here are defaults for training. We can adjust them if we disagree w the huggingface defaults
        default_train_kwargs = {
            "output_dir": output_dir,
        }

        train_kwargs = request.model_dump(exclude_unset=True)
        return {**default_train_kwargs, **(train_kwargs or {})}

    def process_training_args(self, train_kwargs: dict) -> tuple[dict, dict]:
        # NOTE: These also need to be in the GRPOConfigRequest
        trl_keys = [
            "output_dir",
            "temperature",
            "beta",
            "num_iterations",
            "num_generations",
            "per_device_train_batch_size",
            "learning_rate",
            "gradient_accumulation_steps",
            "gradient_checkpointing",
            "lr_scheduler_type",
            "max_prompt_length",
            "max_completion_length",
            "gradient_checkpointing_kwargs",
            "bf16",
            "scale_rewards",
            "max_grad_norm",
        ]
        trl_train_kwargs = {
            key: train_kwargs[key] for key in trl_keys if key in train_kwargs
        }

        arbor_keys = ["update_interval", "lora"]
        arbor_train_kwargs = {
            key: train_kwargs[key] for key in arbor_keys if key in train_kwargs
        }

        return trl_train_kwargs, arbor_train_kwargs

    def initialize(
        self, request: GRPOConfigRequest, inference_manager: InferenceManager
    ):
        """Initialize the training process with ZMQ-based communication."""
        self.train_kwargs = self.find_training_args(request)

        trl_train_kwargs, arbor_train_kwargs = self.process_training_args(
            self.train_kwargs
        )

        self.current_model = request.model

        # Initialize ZMQ socket manager - no need for connection acceptance thread anymore
        self.server_comms_handler = ArborServerCommsHandler()

        script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
        script_path = os.path.join(script_dir, "grpo_training.py")

        # Start the training process with ZMQ ports
        my_env = os.environ.copy()
        my_env["CUDA_VISIBLE_DEVICES"] = self.settings.arbor_config.training.gpu_ids

        num_processes = self.settings.arbor_config.training.gpu_ids.count(",") + 1

        # This is the port for the accelerate main process
        main_process_port = get_free_port()

        params = [
            "python",
            "-m",
            "accelerate.commands.launch",
            "--num_processes",
            str(num_processes),
            "--main_process_port",
            str(main_process_port),
        ]
        if self.settings.arbor_config.training.accelerate_config:
            params.extend(
                [
                    "--config_file",
                    self.settings.arbor_config.training.accelerate_config,
                ]
            )
        params.extend(
            [
                script_path,
                # Comms args
                "--host",
                self.server_comms_handler.host,
                "--command_port",
                str(self.server_comms_handler.command_port),
                "--status_port",
                str(self.server_comms_handler.status_port),
                "--data_port",
                str(self.server_comms_handler.data_port),
                "--broadcast_port",
                str(self.server_comms_handler.broadcast_port),
                "--handshake_port",
                str(self.server_comms_handler.handshake_port),
                # Training args
                "--model",
                self.current_model,
                "--trl_train_kwargs",
                json.dumps(trl_train_kwargs),
                "--arbor_train_kwargs",
                json.dumps(arbor_train_kwargs),
            ]
        )
        print(f"Running following command\n: {' '.join(params)}")

        self.training_process = subprocess.Popen(
            params,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=my_env,
        )

        # A threading.Event to control printing after the server is ready.
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
                        print(f"[GRPO LOG] {line}", end="")

        # Start a background thread to read from the process continuously
        thread = threading.Thread(
            target=_tail_process,
            args=(self.training_process, logs_buffer, stop_printing_event),
            daemon=True,
        )
        thread.start()

        # Start status handling thread
        self.status_thread = threading.Thread(
            target=self._handle_status_updates, args=(inference_manager,), daemon=True
        )
        self.status_thread.start()
        self.server_comms_handler.wait_for_clients(num_processes)

        # Launch the inference server
        print("Launching inference server...")
        inference_manager.launch(self.current_model)

    def _handle_status_updates(self, inference_manager: InferenceManager):
        """Handle status updates from training process using ZMQ SUB socket"""
        print("Starting status update handler...")
        try:

            for status in self.server_comms_handler.receive_status():
                print(f"Received status update: {status}")
                if status["status"] == "model_saved":
                    print("Updating inference model...")
                    # There is a case where this status is sent multiple times
                    # We need to make sure we only update the model once
                    if self._should_update_model():
                        inference_manager.update_model(status["output_dir"])
                        # self.last_inference_update = self.data_count
                        self.model_saved_and_reload_requested = False
                        self.current_model = status["output_dir"]
                        print("Model update complete")
                elif status["status"] == "error":
                    print(f"Training error: {status.get('error', 'Unknown error')}")
                elif status["status"] == "terminated":
                    print("Training process terminated")
                    break
        except Exception as e:
            print(f"Error in status update handler: {e}")

    def grpo_step(
        self, request: GRPORequest, inference_manager: InferenceManager
    ) -> str:
        while inference_manager.is_server_restarting():
            print("Inferece manager restarting, waiting for GRPO step")
            time.sleep(5)

        while self._should_update_model():
            print(
                f"Waiting for model update. Data count: {self.data_count}, Last inference update: {self.last_inference_update}"
            )
            time.sleep(5)

        try:
            # Send the batch to the training process
            self.server_comms_handler.send_data(request.batch)
            self.data_count += 1
        except Exception as e:
            print(f"Failed to send batch to training process: {e}")

        # We tell the script to save the model. The script will let us know when it's done via the status update handler
        # Then we'll actually run the update_model function in the inference manager and finally update the last_inference_update variable
        # if self._should_update_model():
        #     self.server_comms_handler.send_command({"command": "save_model"})

        return self.current_model

    def update_model(self, request, inference_manager: InferenceManager):
        if inference_manager._session:
            # Create a new event loop if one doesn't exist
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Run the session closure in the event loop
            loop.run_until_complete(inference_manager._session.close())
            inference_manager._session = None

        inference_manager.inference_count = 0
        inference_manager.restarting = True

        self.model_saved_and_reload_requested = True
        self.server_comms_handler.send_command({"command": "save_model"})
        while self.model_saved_and_reload_requested:
            print(
                "Waiting for model to be saved and reloaded... This usually takes 20-30 seconds"
            )
            time.sleep(5)
        return self.current_model

    def terminate(self, inference_manager: InferenceManager):
        """Clean up resources and save the final model."""
        try:
            # Stop the inference server
            if inference_manager.process is not None:
                inference_manager.kill()

            # Send termination command through REQ socket
            # self.server_comms_handler.send_broadcast({"message": "terminate"})
            self.training_process.terminate()
            print("Waiting for training process to finish")

            # Wait for training process to finish
            if self.training_process:
                self.training_process.wait(timeout=30)

        except Exception as e:
            print(f"Error during termination: {e}")
        finally:
            # Clean up ZMQ connections
            if self.server_comms_handler:
                self.server_comms_handler.close()

            # Force kill training process if still running
            if self.training_process and self.training_process.poll() is None:
                self.training_process.kill()
                self.training_process.wait()

            # Reinitialize incase we want to start a new training run
            self.training_process = None
            self.current_model = None
            self.server_comms_handler = None
            self.status_thread = None
            self.model_saved_and_reload_requested = False

            self.data_count = 0
            self.last_inference_update = 0

            if self.train_kwargs and "output_dir" in self.train_kwargs:
                print(
                    f"Training completed. Model saved to {self.train_kwargs['output_dir']}"
                )
                if not os.path.exists(self.train_kwargs["output_dir"]):
                    print(
                        f"Warning: Output directory {self.train_kwargs['output_dir']} does not exist"
                    )
                output_dir = self.train_kwargs["output_dir"]
                self.train_kwargs = None
                return output_dir
            else:
                print("Training terminated, no output directory specified")
                self.train_kwargs = None
                return None

    def _should_update_model(self):
        # return (
        #     self.data_count - self.last_inference_update
        #     >= self.train_kwargs["update_interval"]
        # )
        return self.model_saved_and_reload_requested


def get_free_port() -> int:
    """
    Return a free TCP port on localhost.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]
