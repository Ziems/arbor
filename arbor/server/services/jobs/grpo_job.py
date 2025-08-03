import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import coolname
import psutil

from arbor.server.api.models.schemas import (
    GRPOCheckpointRequest,
    GRPOInitializeRequest,
    GRPOStatus,
    GRPOStepRequest,
)
from arbor.server.core.config import Config
from arbor.server.services.comms.comms import ArborServerCommsHandler
from arbor.server.services.jobs.inference_job import InferenceJob
from arbor.server.services.jobs.inference_launch_config import InferenceLaunchConfig
from arbor.server.services.jobs.job import Job
from arbor.server.services.managers.inference_manager import InferenceManager
from arbor.server.utils.helpers import get_free_port
from arbor.server.utils.logging import get_logger

logger = get_logger(__name__)


class GRPOJob(Job):
    def __init__(self, config: Config, request: GRPOInitializeRequest):
        id = self._make_job_id(request)
        super().__init__(id=id)
        self.config = config
        self.trainin_process = None
        self.base_model = None
        self.train_kwargs = None
        self.server_comms_handler = None
        self.status_thread = None
        self.saving_checkpoint = False
        self.saving_model = False
        self.terminating = False
        self.inference_job: InferenceJob = None

        self.checkpoints = {}
        self.last_checkpoint = None
        self.data_count = 0
        self.last_inference_update = 0

    def _make_job_id(self, request: GRPOInitializeRequest):
        slug = coolname.generate_slug(2)
        model = request.model.split("/")[-1].lower()
        suffix = request.suffix if request.suffix is not None else slug
        timestamp = datetime.now().strftime("%Y%m%d")
        return f"grpo:{model}:{suffix}:{timestamp}"

    def _make_output_dir(self, request: GRPOInitializeRequest):
        return str(Path(self.config.STORAGE_PATH).resolve() / "models" / self.id)

    def find_training_args(self, request: GRPOInitializeRequest) -> dict:
        """Process the config request and return training arguments."""
        output_dir = self._make_output_dir(request)

        # Here are defaults for training. We can adjust them if we disagree w the huggingface defaults
        default_train_kwargs = {"output_dir": output_dir, "grpo_flavor": "grpo"}

        train_kwargs = request.model_dump(exclude_unset=True)
        return {**default_train_kwargs, **(train_kwargs or {})}

    def process_training_args(
        self, train_kwargs: GRPOInitializeRequest
    ) -> tuple[dict, dict]:
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

        arbor_keys = ["max_context_length", "lora", "wandb_kwargs", "grpo_flavor"]
        arbor_train_kwargs = {
            key: train_kwargs[key] for key in arbor_keys if key in train_kwargs
        }

        return trl_train_kwargs, arbor_train_kwargs

    def initialize(
        self, request: GRPOInitializeRequest, inference_manager: InferenceManager
    ):
        """Initialize the training process with ZMQ-based communication."""
        self.train_kwargs = self.find_training_args(request)
        trl_train_kwargs, arbor_train_kwargs = self.process_training_args(
            self.train_kwargs
        )

        inference_launch_config = InferenceLaunchConfig(
            max_context_length=arbor_train_kwargs.get("max_context_length", None),
            # TODO: `gpu_ids` is hardcoded and may be better to be passed in the request
            # Likely needs to be the number of GPUs and not the ids themselves
            # So this would be best to do once we have a Resource manager
            gpu_ids=self.config.arbor_config.inference.gpu_ids,
            is_grpo=True,
            grpo_job_id=self.id,
        )
        logger.info("Launching inference server...")
        self.inference_job = inference_manager.launch_job(
            request.model,
            inference_launch_config,
        )

        # Initialize ZMQ socket manager - no need for connection acceptance thread anymore
        self.server_comms_handler = ArborServerCommsHandler()

        script_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"
        )
        script_name = {"mmgrpo": "mmgrpo_training.py", "grpo": "grpo_training.py"}[
            arbor_train_kwargs["grpo_flavor"]
        ]
        script_path = os.path.join(script_dir, script_name)

        # Start the training process with ZMQ ports
        my_env = os.environ.copy()
        # Convert gpu_ids list to comma-separated string for environment variable
        gpu_ids_str = ",".join(map(str, self.config.arbor_config.training.gpu_ids))
        my_env["CUDA_VISIBLE_DEVICES"] = gpu_ids_str
        # WandB can block the training process for login, so we silence it
        my_env["WANDB_SILENT"] = "true"

        num_processes = len(self.config.arbor_config.training.gpu_ids)

        # This is the port for the accelerate main process
        main_process_port = get_free_port()

        params = [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            "--num_processes",
            str(num_processes),
            "--main_process_port",
            str(main_process_port),
        ]
        if self.config.arbor_config.training.accelerate_config:
            params.extend(
                [
                    "--config_file",
                    self.config.arbor_config.training.accelerate_config,
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
                str(self.inference_job.port),
                "--vllm_group_port",
                str(self.inference_job.group_port),
                # Training args
                "--model",
                request.model,
                "--trl_train_kwargs",
                json.dumps(trl_train_kwargs),
                "--arbor_train_kwargs",
                json.dumps(arbor_train_kwargs),
            ]
        )
        logger.info(f"Running GRPO training command: {' '.join(params)}")

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
                    # Log only if stop_event is not set
                    if not stop_event.is_set():
                        logger.info(f"[GRPO LOG] {line.strip()}")

        # Start a background thread to read from the process continuously
        thread = threading.Thread(
            target=_tail_process,
            args=(self.training_process, logs_buffer, stop_printing_event),
            daemon=True,
        )
        thread.start()

        # Start status handling thread
        self.status_thread = threading.Thread(
            target=self._handle_status_updates, args=(), daemon=True
        )
        self.status_thread.start()
        self.server_comms_handler.wait_for_clients(num_processes)

    def _handle_status_updates(self):
        """Handle status updates from training process using ZMQ SUB socket"""
        logger.info("Starting status update handler...")
        try:
            for status in self.server_comms_handler.receive_status():
                logger.debug(f"Received status update: {status}")
                if status["status"] == "weight_update_start":
                    # Block inference calls by incrementing counter
                    self.inference_job.start_weight_update()
                elif status["status"] == "weight_update_complete":
                    # Decrement counter to potentially allow inference calls again
                    self.inference_job.complete_weight_update()
                elif status["status"] == "model_saved":
                    logger.info("Updating inference model...")
                    # There is a case where this status is sent multiple times
                    # We need to make sure we only update the model once
                    self.saving_model = False
                    logger.info("Model update complete")
                elif status["status"] == "checkpoint_saved":
                    logger.info("Received checkpoint saved status")
                    self.checkpoints[status["checkpoint_name"]] = status["output_dir"]
                    self.last_checkpoint = status["checkpoint_name"]
                    self.saving_checkpoint = False
                    logger.info("Checkpoint saved")
                elif status["status"] == "error":
                    error_msg = status.get("error", "Unknown error")
                    logger.error(f"Training error: {error_msg}")
                elif status["status"] == "terminated":
                    self.terminating = False
                    logger.info("Training process terminated")
        except Exception as e:
            logger.error(f"Error in status update handler: {e}")
            # Make sure to allow inference if there's an error
            try:
                self.inference_job.complete_weight_update()
            except:
                pass

    def validate_batch(self, batch):
        if not isinstance(batch, list):
            raise ValueError("Batch must be a list")

        if self.train_kwargs["grpo_flavor"] == "mmgrpo":
            for group in batch:
                if not isinstance(group, list):
                    raise ValueError("Each group in batch must be a list")
                for item in group:
                    if not isinstance(item, dict):
                        raise ValueError("Each item in group must be a dictionary")
                    required_keys = {"messages", "completion", "advantage"}
                    if not all(key in item for key in required_keys):
                        raise ValueError(
                            f"Each item must contain keys: {required_keys}"
                        )
            return True
        elif self.train_kwargs["grpo_flavor"] == "grpo":
            for item in batch:
                if not isinstance(item, dict):
                    raise ValueError("Each item in batch must be a dictionary")
                required_keys = {"messages", "completion", "reward"}
                if not all(key in item for key in required_keys):
                    raise ValueError(f"Each item must contain keys: {required_keys}")
            return True
        else:
            raise NotImplementedError(
                f"GRPO flavor batch validation not implemented for {self.train_kwargs['grpo_flavor']}"
            )

    def grpo_step(self, request: GRPOStepRequest) -> str:
        while self.saving_checkpoint:
            logger.info(
                "Saving checkpoint, pausing GRPO steps until checkpoint is saved..."
            )
            time.sleep(5)

        self.validate_batch(request.batch)

        try:

            # Send the batch to the training process
            self.server_comms_handler.send_data(request.batch)
            self.data_count += 1

        except Exception as e:
            logger.error(f"Failed to send batch to training process: {e}")
            raise

    def checkpoint(self, request: GRPOCheckpointRequest):
        while (
            self.inference_job.is_updating
        ):  # Use the property instead of direct access
            logger.info("Waiting for weight updates to finish before checkpointing...")
            time.sleep(5)

        self.saving_checkpoint = True
        self.server_comms_handler.send_command(
            {"command": "save_checkpoint", "checkpoint_name": request.checkpoint_name}
        )
        while self.saving_checkpoint:
            logger.info("Waiting for checkpoint to be saved...")
            time.sleep(5)

    def terminate(self):
        """Clean up resources and save the final model."""
        time.sleep(5)

        while (
            self.inference_job and self.inference_job.is_updating
        ):  # Use the property instead of direct access
            logger.info("Waiting for final weight updates to finish before saving...")
            time.sleep(5)

        logger.info("Sending save model command")
        self.saving_model = True
        self.server_comms_handler.send_command({"command": "save_model"})
        while self.saving_model:
            logger.info("Waiting for final model to be saved...")
            time.sleep(5)

        logger.info("Sending termination command")
        self.terminating = True
        self.server_comms_handler.send_command({"command": "terminate"})
        logger.info("Waiting for training process to finish...")

        # Wait for at most 15 seconds for termination
        start_time = time.time()
        while self.terminating:
            if time.time() - start_time > 15:
                logger.warning(
                    "Termination wait timed out after 15 seconds, proceeding with cleanup..."
                )
                break
            logger.info("Waiting for run to be terminated...")
            time.sleep(3)

        logger.info("Starting cleanup")
        self.cleanup_termination()

        if self.train_kwargs and "output_dir" in self.train_kwargs:
            output_dir = self.train_kwargs["output_dir"]
            logger.info(f"Training completed. Model saved to {output_dir}")
            logger.info(f"Training logs and checkpoints are stored in: {output_dir}")
            if not os.path.exists(output_dir):
                logger.warning(f"Output directory {output_dir} does not exist")
            self.train_kwargs = None
        else:
            logger.info("Training terminated, no output directory specified")
            self.train_kwargs = None

    def cleanup_termination(self):
        try:
            # Kill training process and all its children (accelerate launcher creates multiple processes)
            if self.training_process:
                logger.info("Terminating training process and its children...")
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
                    logger.warning(f"Process {self.training_process.pid} not found")
                except Exception as e:
                    logger.error(f"Error killing training process tree: {e}")
                    # Fallback to basic termination
                    self.training_process.terminate()
                    try:
                        self.training_process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self.training_process.kill()
                        self.training_process.wait(timeout=10)

            # Clean up ZMQ connections
            if self.server_comms_handler:
                logger.debug("Closing ZMQ connections...")
                self.server_comms_handler.close()

            if self.inference_job and self.inference_job.process is not None:
                logger.info("Killing inference manager...")
                self.inference_job.kill()

            # Reinitialize in case we want to start a new training run
            self.training_process = None
            self.server_comms_handler = None
            self.status_thread = None
            self.data_count = 0
            logger.info("Cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            # Still reset state even if cleanup fails
            self.training_process = None
            self.server_comms_handler = None
            self.status_thread = None
            self.data_count = 0

    def get_status(self) -> GRPOStatus:
        return GRPOStatus(
            job_id=self.id,
            status=self.status.value,
            current_model=self.id,
            checkpoints=self.checkpoints,
            last_checkpoint=self.last_checkpoint,
        )
