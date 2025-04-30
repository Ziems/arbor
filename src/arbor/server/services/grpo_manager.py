import random
import string
import time
import json
from datetime import datetime
from pathlib import Path
import os
import subprocess
import threading
from typing import Optional
import signal
import sys

from arbor.server.api.models.schemas import GRPORequest, GRPOConfigRequest
from arbor.server.core.config import Settings
from arbor.server.services.inference_manager import InferenceManager
from arbor.server.services.comms.comms import ArborServerCommsHandler


class GRPOManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.training_process = None
        self.current_model = None
        self.train_kwargs = None
        self.server_comms_handler = None
        self.status_thread = None

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

        # TODO: Here are defaults for training. We can adjust them if we disagree w the huggingface defaults
        default_train_kwargs = {
            # "use_peft": False,
            # "num_train_epochs": 5,
            # "per_device_train_batch_size": 1,
            # "gradient_accumulation_steps": 8,
            # "learning_rate": 1e-5,
            # "max_seq_length": None,
            # "packing": False, # TODO: Turning this off for now
            # "bf16": True,
            "output_dir": output_dir,
            # "beta": 0.04,
            # "num_iterations": 1,
            # "temperature": 0.9,
            # "num_generations": 8,
            # "scale_rewards": True,
            # "epsilon_low": 0.2,
            # "epsilon_high": 0.2,
            # "update_interval": 25
        }

        train_kwargs = request.model_dump(exclude_unset=True)
        return {**default_train_kwargs, **(train_kwargs or {})}

    def process_training_args(self, train_kwargs: dict) -> tuple[dict, dict]:
        trl_keys = [
            "output_dir",
            "temperature",
            "beta",
            "num_iterations",
            "num_generations",
        ]
        trl_train_kwargs = {
            key: train_kwargs[key] for key in trl_keys if key in train_kwargs
        }

        arbor_keys = ["update_interval"]
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

        params = [
            "python",
            "-m",
            "accelerate.commands.launch",
            "--num_processes",
            str(self.settings.arbor_config.training.num_processes),
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
            # Training args
            "--model",
            self.current_model,
            "--trl_train_kwargs",
            json.dumps(trl_train_kwargs),
            "--arbor_train_kwargs",
            json.dumps(arbor_train_kwargs),
        ]
        print(f"Running following command\n: {' '.join(params)}")

        self.training_process = subprocess.Popen(
            params,
            env=my_env,
        )

        # Start status handling thread
        self.status_thread = threading.Thread(
            target=self._handle_status_updates, args=(inference_manager,), daemon=True
        )
        self.status_thread.start()

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
                        self.last_inference_update = self.data_count
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
        if self._should_update_model():
            self.server_comms_handler.send_command({"command": "save_model"})

        return self.current_model

    def terminate(self, inference_manager: InferenceManager):
        """Clean up resources and save the final model."""
        try:
            # Stop the inference server
            if inference_manager.process is not None:
                inference_manager.kill()

            # Send termination command through REQ socket
            self.server_comms_handler.send_broadcast({"message": "terminate"})

            # Wait for training process to finish
            if self.training_process:
                self.training_process.wait(timeout=30)

        except Exception as e:
            print(f"Error during termination: {e}")
        finally:
            # Clean up ZMQ connections
            if self.server_comms_handler:
                self.server_comms_handler.close()

            if self.train_kwargs and "output_dir" in self.train_kwargs:
                print(
                    f"Training completed. Model saved to {self.train_kwargs['output_dir']}"
                )
                if not os.path.exists(self.train_kwargs["output_dir"]):
                    print(
                        f"Warning: Output directory {self.train_kwargs['output_dir']} does not exist"
                    )
                return self.train_kwargs["output_dir"]
            else:
                print("Training terminated, no output directory specified")
                return None

    def _should_update_model(self):
        return (
            self.data_count - self.last_inference_update
            >= self.train_kwargs["update_interval"]
        )
