import json
import os
import random
import socket
import subprocess
import sys
import threading

from arbor.server.api.models.schemas import (
    FineTuneRequest,
)
from arbor.server.core.config import Settings
from arbor.server.services.comms.comms import ArborServerCommsHandler
from arbor.server.services.managers.file_manager import FileManager
from arbor.server.utils.logging import get_logger

logger = get_logger(__name__)


class FileTrainJob:
    def __init__(self, settings: Settings):
        self.settings = settings

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

        name, output_dir = self.make_output_dir(request)

        return data_path, output_dir

    def find_train_args_sft(self, request: FineTuneRequest, file_manager: FileManager):
        data_path, output_dir = self._prepare_training_file(
            request, file_manager, "sft"
        )

        default_train_kwargs = {
            "use_peft": False,
            "num_train_epochs": 5,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 1e-5,
            "max_seq_length": None,
            "packing": True,
            "bf16": True,
            "output_dir": output_dir,
            "train_data_path": data_path,
        }

        train_kwargs = {"packing": False}
        train_kwargs = {**default_train_kwargs, **(train_kwargs or {})}

        return train_kwargs

    def find_train_args_dpo(self, request: FineTuneRequest, file_manager: FileManager):
        data_path, output_dir = self._prepare_training_file(
            request, file_manager, "dpo"
        )

        default_train_kwargs = {
            "use_peft": False,
            "num_train_epochs": 5,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 1e-5,
            "max_seq_length": None,
            "packing": True,
            "bf16": True,
            "output_dir": output_dir,
            "train_data_path": data_path,
        }

        train_kwargs = {"packing": False}
        train_kwargs = {**default_train_kwargs, **(train_kwargs or {})}

        return train_kwargs

    def fine_tune(self, request: FineTuneRequest, file_manager: FileManager):

        train_type = request.method["type"]

        args_fn = {
            "dpo": self.find_train_args_dpo,
            "sft": self.find_train_args_sft,
        }[train_type]

        trl_train_kwargs, arbor_train_kwargs = args_fn(request, file_manager)

        self.model = request.model

        self.server_comms_handler = ArborServerCommsHandler()

        script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
        script_name = {"dpo": "dpo_training.py", "sft": "sft_training.py"}[train_type]
        script_path = os.path.join(script_dir, script_name)

        my_env = os.environ.copy()
        my_env["CUDA_VISIBLE_DEVICES"] = self.settings.arbor_config.training.gpu_ids
        # WandB can block the training process for login, so we silence it
        my_env["WANDB_SILENT"] = "true"

        num_processes = self.settings.arbor_config.training.gpu_ids.count(",") + 1
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

    def _handle_status_updates(self):
        for status in self.server_comms_handler.receive_status():
            logger.debug(f"Received status update: {status}")

    def terminate(self):
        raise NotImplementedError("Not implemented")


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
            logger.error(f"Error binding to port: {e}")
    return random.choice(ports)
