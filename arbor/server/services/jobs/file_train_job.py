import json
import os
import random
import string
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Literal

from arbor.server.api.models.schemas import (
    FineTuneRequest,
    JobStatusModel,
)
from arbor.server.core.config import Config
from arbor.server.services.comms.comms import ArborServerCommsHandler
from arbor.server.services.jobs.job import Job
from arbor.server.services.managers.file_manager import FileManager
from arbor.server.utils.helpers import get_free_port
from arbor.server.utils.logging import get_logger
from arbor.server.utils.mock_utils import get_script_path, setup_mock_environment

logger = get_logger(__name__)


class FileTrainJob(Job):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.model = None
        self.training_file = None
        self.fine_tuned_model = None

    def _make_output_dir(self, request: FineTuneRequest):
        model_name = request.model.split("/")[-1].lower()
        suffix = (
            request.suffix
            if request.suffix is not None
            else "".join(random.choices(string.ascii_letters + string.digits, k=6))
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"ft:{model_name}:{suffix}:{timestamp}"
        return name, str(Path(self.config.STORAGE_PATH).resolve() / "models" / name)

    def _prepare_training_file(
        self, request: FineTuneRequest, file_manager: FileManager, format_type: str
    ):
        """
        Common logic for file validation and setup for training methods.

        Args:
            request: The fine-tune request
            file_manager: The file manager instance
            format_type: Format type to validate ('sft' or 'dpo')

        Returns:
            tuple: (data_path, output_dir)
        """
        file = file_manager.get_file(request.training_file)
        if file is None:
            raise ValueError(f"Training file {request.training_file} not found")

        data_path = file["path"]

        # Validate file format using the unified method
        file_manager.validate_file_format(data_path, format_type)

        return data_path

    def find_train_args_sft(self, request: FineTuneRequest, file_manager: FileManager):
        name, output_dir = self._make_output_dir(request)
        data_path = self._prepare_training_file(request, file_manager, "sft")

        default_train_kwargs = {
            "num_train_epochs": 5,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 1e-5,
            "max_seq_length": None,
            "packing": True,
            "bf16": True,
            "output_dir": output_dir,
        }

        train_kwargs = {"packing": False}
        train_kwargs = {**default_train_kwargs, **(train_kwargs or {})}

        arbor_train_kwargs = {
            "train_data_path": data_path,
            "lora": False,
        }

        return train_kwargs, arbor_train_kwargs

    def find_train_args_dpo(self, request: FineTuneRequest, file_manager: FileManager):
        name, output_dir = self._make_output_dir(request)
        data_path = self._prepare_training_file(request, file_manager, "dpo")

        default_train_kwargs = {
            "num_train_epochs": 5,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 1e-5,
            "max_seq_length": None,
            "packing": True,
            "bf16": True,
            "output_dir": output_dir,
        }

        train_kwargs = {"packing": False}
        train_kwargs = {**default_train_kwargs, **(train_kwargs or {})}

        arbor_train_kwargs = {
            "train_data_path": data_path,
            "lora": False,
        }

        return train_kwargs, arbor_train_kwargs

    def fine_tune(
        self,
        request: FineTuneRequest,
        file_manager: FileManager,
        train_type: Literal["dpo", "sft"],
    ):

        find_train_args_fn = {
            "dpo": self.find_train_args_dpo,
            "sft": self.find_train_args_sft,
        }[train_type]
        trl_train_kwargs, arbor_train_kwargs = find_train_args_fn(request, file_manager)

        self.model = request.model
        self.training_file = request.training_file
        print("Set model and training file")

        self.server_comms_handler = ArborServerCommsHandler()

        script_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"
        )
        script_name = {"dpo": "dpo_training.py", "sft": "sft_training.py"}[train_type]
        script_path = get_script_path(script_name, script_dir)

        my_env = os.environ.copy()
        # TODO: This should first check to see if GPUs are available w/ a resource manager or something
        # Convert gpu_ids list to comma-separated string for environment variable
        gpu_ids_str = ",".join(map(str, self.config.arbor_config.training.gpu_ids))
        my_env["CUDA_VISIBLE_DEVICES"] = gpu_ids_str
        # WandB can block the training process for login, so we silence it
        my_env["WANDB_SILENT"] = "true"

        # Setup mock environment if needed
        my_env = setup_mock_environment(my_env)

        num_processes = len(self.config.arbor_config.training.gpu_ids)
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
                # Training args
                "--model",
                self.model,
                "--trl_train_kwargs",
                json.dumps(trl_train_kwargs),
                "--arbor_train_kwargs",
                json.dumps(arbor_train_kwargs),
            ]
        )
        logger.info(f"Running training command: {' '.join(params)}")

        self.training_process = subprocess.Popen(
            params,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=my_env,
        )

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
                        logger.info(f"[{train_type.upper()} LOG] {line.strip()}")

        thread = threading.Thread(
            target=_tail_process,
            args=(self.training_process, logs_buffer, stop_printing_event),
            daemon=True,
        )
        thread.start()
        self.server_comms_handler.wait_for_clients(num_processes)

    def _handle_status_updates(self):
        for status in self.server_comms_handler.receive_status():
            logger.debug(f"Received status update: {status}")

    def terminate(self):
        """Terminate the training process and clean up resources"""
        logger.info(f"Terminating FileTrainJob {self.id}")

        # Terminate the training process if it exists
        if hasattr(self, "training_process") and self.training_process:
            try:
                logger.info(f"Terminating training process {self.training_process.pid}")
                # Try graceful termination first
                self.training_process.terminate()

                # Wait a bit for graceful shutdown
                try:
                    self.training_process.wait(timeout=5)
                    logger.info("Training process terminated gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't terminate gracefully
                    logger.warning(
                        "Training process didn't terminate gracefully, force killing"
                    )
                    self.training_process.kill()
                    self.training_process.wait()
                    logger.info("Training process force killed")

            except Exception as e:
                logger.error(f"Error terminating training process: {e}")

        # Clean up comms handler if it exists
        if hasattr(self, "server_comms_handler") and self.server_comms_handler:
            try:
                # Assuming the comms handler might have cleanup methods
                if hasattr(self.server_comms_handler, "cleanup"):
                    self.server_comms_handler.cleanup()
                elif hasattr(self.server_comms_handler, "close"):
                    self.server_comms_handler.close()
            except Exception as e:
                logger.error(f"Error cleaning up comms handler: {e}")

        logger.info(f"FileTrainJob {self.id} termination completed")

    def to_status_model(self) -> JobStatusModel:
        print("To status model", self.model, self.training_file, self.fine_tuned_model)
        return JobStatusModel(
            id=self.id,
            status=self.status.value,
            model=self.model,
            training_file=self.training_file,
            fine_tuned_model=self.fine_tuned_model,
        )
