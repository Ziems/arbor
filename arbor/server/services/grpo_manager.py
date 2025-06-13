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

import psutil

from arbor.server.api.models.schemas import (
    GRPOCheckpointRequest,
    GRPOConfigRequest,
    GRPORequest,
)
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
        self.saving_checkpoint = False
        self.saving_model = False
        self.terminating = False

        self.checkpoints = {}
        self.last_checkpoint = None
        self.data_count = 0
        self.last_inference_update = 0
        # Set up signal handler
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle keyboard interrupt (SIGINT) gracefully."""
        print("\nReceived keyboard interrupt. Shutting down gracefully...")
        # Sleep for a bit to let async operations go through
        time.sleep(2)
        if self.training_process is not None:
            self.cleanup_termination(None)

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
            "report_to",
            "log_completions",
            "logging_steps",
            "generation_batch_size",
            "mask_truncated_completions",
        ]
        trl_train_kwargs = {
            key: train_kwargs[key] for key in trl_keys if key in train_kwargs
        }

        arbor_keys = ["max_context_length", "lora"]
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

        # The inference server has to be launched before the training process
        # Launch the inference server
        # launch_kwargs = {
        #     k: v for k, v in arbor_train_kwargs.items() if k in ["max_context_length"]
        # }
        inference_manager.launch_kwargs["max_context_length"] = arbor_train_kwargs.get(
            "max_context_length", None
        )
        print("Launching inference server...")
        inference_manager.launch(self.current_model)

        # Initialize ZMQ socket manager - no need for connection acceptance thread anymore
        self.server_comms_handler = ArborServerCommsHandler()

        script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
        script_path = os.path.join(script_dir, "grpo_training.py")

        # Start the training process with ZMQ ports
        my_env = os.environ.copy()
        my_env["CUDA_VISIBLE_DEVICES"] = self.settings.arbor_config.training.gpu_ids
        # WandB can block the training process for login, so we silence it
        my_env["WANDB_SILENT"] = "true"

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
                "--vllm_port",
                str(inference_manager.port),
                "--vllm_group_port",
                str(inference_manager.group_port),
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

    async def _handle_weight_update_start(self, inference_manager):
        """Handle weight update start in the event loop"""
        await inference_manager.start_weight_update()

    async def _handle_weight_update_complete(self, inference_manager):
        """Handle weight update complete in the event loop"""
        await inference_manager.complete_weight_update()

    def _run_in_loop(self, coro):
        """Run a coroutine in the event loop from a thread"""
        future = asyncio.run_coroutine_threadsafe(coro, self.event_loop)
        return future.result()

    def _handle_status_updates(self, inference_manager: InferenceManager):
        """Handle status updates from training process using ZMQ SUB socket"""
        print("Starting status update handler...")
        try:
            for status in self.server_comms_handler.receive_status():
                print(f"Received status update: {status}")
                if status["status"] == "weight_update_start":
                    # Block inference calls by incrementing counter
                    inference_manager.start_weight_update()
                elif status["status"] == "weight_update_complete":
                    # Decrement counter to potentially allow inference calls again
                    inference_manager.complete_weight_update()
                elif status["status"] == "model_saved":
                    print("Updating inference model...")
                    # There is a case where this status is sent multiple times
                    # We need to make sure we only update the model once
                    self.saving_model = False
                    print("Model update complete")
                elif status["status"] == "checkpoint_saved":
                    print("Received checkpoint saved status")
                    self.checkpoints[status["checkpoint_name"]] = status["output_dir"]
                    self.last_checkpoint = status["checkpoint_name"]
                    self.saving_checkpoint = False
                    print("Checkpoint saved")
                elif status["status"] == "error":
                    print(f"Training error: {status.get('error', 'Unknown error')}")
                elif status["status"] == "terminated":
                    self.terminating = False
                    print("Training process terminated")
        except Exception as e:
            print(f"Error in status update handler: {e}")
            # Make sure to allow inference if there's an error
            try:
                inference_manager.complete_weight_update()
            except:
                pass

    def grpo_step(
        self, request: GRPORequest, inference_manager: InferenceManager
    ) -> str:
        while self.saving_checkpoint:
            print("Saving checkpoint, pausing GRPO steps until checkpoint is saved...")
            time.sleep(5)

        try:
            # Send the batch to the training process
            self.server_comms_handler.send_data(request.batch)
            self.data_count += 1

        except Exception as e:
            print(f"Failed to send batch to training process: {e}")
            raise

        self.current_model = self.train_kwargs["output_dir"]
        inference_manager.launched_model = self.current_model

        return {
            "current_model": self.current_model,
            "checkpoints": self.checkpoints,
            "last_checkpoint": self.last_checkpoint,
        }

    def checkpoint(
        self, request: GRPOCheckpointRequest, inference_manager: InferenceManager
    ):
        while (
            inference_manager.is_updating
        ):  # Use the property instead of direct access
            print("Waiting for weight updates to finish before checkpointing...")
            time.sleep(5)

        self.saving_checkpoint = True
        self.server_comms_handler.send_command(
            {"command": "save_checkpoint", "checkpoint_name": request.checkpoint_name}
        )
        while self.saving_checkpoint:
            print("Waiting for checkpoint to be saved...")
            time.sleep(5)
        return {
            "current_model": self.current_model,
            "checkpoints": self.checkpoints,
            "last_checkpoint": self.last_checkpoint,
        }

    def terminate(self, inference_manager: InferenceManager):
        """Clean up resources and save the final model."""
        time.sleep(5)

        while (
            inference_manager and inference_manager.is_updating
        ):  # Use the property instead of direct access
            print("Waiting for final weight updates to finish before saving...")
            time.sleep(5)

        print("sending save model command")
        self.saving_model = True
        self.server_comms_handler.send_command({"command": "save_model"})
        while self.saving_model:
            print("Waiting for final model to be saved...")
            time.sleep(5)

        termination_data = {
            "current_model": self.current_model,
            "checkpoints": self.checkpoints,
            "last_checkpoint": self.last_checkpoint,
        }

        print("sending termination command")
        self.terminating = True
        self.server_comms_handler.send_command({"command": "terminate"})
        print("Waiting for training process to finish...")

        # Wait for at most 15 seconds for termination
        start_time = time.time()
        while self.terminating:
            if time.time() - start_time > 15:
                print(
                    "Termination wait timed out after 15 seconds, proceeding with cleanup..."
                )
                break
            print("Waiting for run to be terminated...")
            time.sleep(3)

        print("Doing cleanup")
        self.cleanup_termination(inference_manager)

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
        else:
            print("Training terminated, no output directory specified")
            self.train_kwargs = None

        return termination_data

    def cleanup_termination(self, inference_manager):
        try:
            # Kill training process and all its children (accelerate launcher creates multiple processes)
            if self.training_process:
                print("Terminating training process and its children...")
                try:
                    parent = psutil.Process(self.training_process.pid)
                    # Get all child processes including grandchildren
                    children = parent.children(recursive=True)

                    # Send SIGTERM to children first
                    for child in children:
                        try:
                            child.send_signal(signal.SIGTERM)
                        except psutil.NoSuchProcess:
                            pass

                    # Send SIGTERM to parent
                    parent.send_signal(signal.SIGTERM)

                    # Wait for processes to terminate gracefully
                    gone, alive = psutil.wait_procs(children + [parent], timeout=10)

                    # If any processes are still alive, force kill them
                    for p in alive:
                        try:
                            p.kill()  # SIGKILL
                        except psutil.NoSuchProcess:
                            pass

                except psutil.NoSuchProcess:
                    print(f"Process {self.training_process.pid} not found")
                except Exception as e:
                    print(f"Error killing training process tree: {e}")
                    # Fallback to basic termination
                    self.training_process.terminate()
                    try:
                        self.training_process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self.training_process.kill()
                        self.training_process.wait(timeout=10)

            # Clean up ZMQ connections
            if self.server_comms_handler:
                print("Closing ZMQ connections...")
                self.server_comms_handler.close()

            if inference_manager and inference_manager.process is not None:
                print("Killing inference manager...")
                inference_manager.kill()

            # Reinitialize in case we want to start a new training run
            self.training_process = None
            self.current_model = None
            self.server_comms_handler = None
            self.status_thread = None
            self.data_count = 0
            print("Cleanup completed successfully")
        except Exception as e:
            print(f"Error during cleanup: {e}")
            # Still reset state even if cleanup fails
            self.training_process = None
            self.current_model = None
            self.server_comms_handler = None
            self.status_thread = None
            self.data_count = 0


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
