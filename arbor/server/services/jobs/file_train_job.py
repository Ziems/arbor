import json
import os
from typing import Literal, Optional

from arbor.server.api.models.schemas import (
    FineTuneRequest,
    JobStatus,
    JobStatusModel,
)
from arbor.server.core.config import Config
from arbor.server.services.managers.gpu_manager import GPUManager
from arbor.server.services.jobs.job import Job, JobArtifact
from arbor.server.services.managers.file_manager import FileManager
from arbor.server.utils.helpers import get_free_port
from arbor.server.utils.logging import get_logger
from arbor.server.utils.mock_utils import get_script_path, setup_mock_environment
from arbor.server.utils.process_runner import AccelerateProcessRunner

logger = get_logger(__name__)


class FileTrainJob(Job):
    def __init__(self, config: Config, gpu_manager: Optional[GPUManager] = None):
        # Training jobs need logs, models, and checkpoints
        super().__init__(
            config,
            artifacts=[JobArtifact.LOGS, JobArtifact.MODEL, JobArtifact.CHECKPOINTS],
        )
        self.gpu_manager = gpu_manager
        self.model = None
        self.training_file = None
        self.fine_tuned_model = None
        self.allocated_gpus = None
        self.process_runner: Optional[AccelerateProcessRunner] = None

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
        output_dir = self._make_model_dir()  # Use base class method
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
        output_dir = self._make_model_dir()  # Use base class method
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
        # Allocate GPUs from GPU manager
        assert self.gpu_manager is not None, "FileTrainJob requires a GPUManager"
        self.allocated_gpus = self.gpu_manager.allocate_gpus(self.id, request.num_gpus)
        logger.info(f"Allocated GPUs {self.allocated_gpus} for FileTrainJob {self.id}")

        find_train_args_fn = {
            "dpo": self.find_train_args_dpo,
            "sft": self.find_train_args_sft,
        }[train_type]
        trl_train_kwargs, arbor_train_kwargs = find_train_args_fn(request, file_manager)

        self.model = request.model
        self.training_file = request.training_file
        print("Set model and training file")

        script_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"
        )
        script_name = {"dpo": "dpo_training.py", "sft": "sft_training.py"}[train_type]
        script_path = get_script_path(script_name, script_dir)

        my_env = os.environ.copy()
        # Use allocated GPUs instead of all config GPUs
        gpu_ids_str = ",".join(map(str, self.allocated_gpus))
        my_env["CUDA_VISIBLE_DEVICES"] = gpu_ids_str

        # Handle WandB configuration
        if trl_train_kwargs.get("report_to") == "wandb":
            # WandB is explicitly requested, just silence login prompts
            my_env["WANDB_SILENT"] = "true"
        else:
            # WandB not requested, disable it completely to avoid login errors
            my_env["WANDB_SILENT"] = "true"
            trl_train_kwargs["report_to"] = "none"

        # Configure ZMQ for better stability and error handling
        my_env["ZMQ_MAX_SOCKETS"] = "1024"
        my_env["ZMQ_IO_THREADS"] = "1"
        # Increase file descriptor limits to prevent resource exhaustion
        my_env["RLIMIT_NOFILE"] = "4096"
        # Set ZMQ socket options for better error handling
        my_env["ZMQ_LINGER"] = "0"

        # Setup mock environment if needed
        my_env = setup_mock_environment(my_env)

        num_processes = len(self.allocated_gpus)
        main_process_port = get_free_port()

        logger.info("Running training command")

        # Use clean process runner for training
        self.process_runner = AccelerateProcessRunner(self.id)

        # Build script args directly (everything that goes after the script path)
        script_args = [
            # Comms args
            "--host",
            self.server_comms_handler.host,
            "--command_port",
            str(self.server_comms_handler.command_port),
            "--event_port",
            str(self.server_comms_handler.event_port),
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

        # Override log file with train type specific name
        log_dir = self._make_log_dir()
        self.log_file_path = os.path.join(log_dir, f"{train_type}_training.log")
        log_callback = self.create_log_callback(train_type.upper())

        self.training_process = self.process_runner.start_training(
            script_path=script_path,
            num_processes=num_processes,
            main_process_port=main_process_port,
            script_args=script_args,
            accelerate_config=self.config.accelerate_config,
            env=my_env,
            log_callback=log_callback,
        )

        self.server_comms_handler.wait_for_clients(num_processes)

        # Set job to running status
        self.status = JobStatus.RUNNING

        # Start monitoring thread for process completion
        import threading

        self.completion_thread = threading.Thread(
            target=self._monitor_completion,
            args=(trl_train_kwargs["output_dir"],),
            daemon=True,
        )
        self.completion_thread.start()

    def _monitor_completion(self, output_dir: str):
        """Monitor training process completion and update job status."""
        try:
            # Wait for process to complete
            exit_code = self.process_runner.wait()

            if exit_code == 0:
                # Training completed successfully
                logger.info(f"Training completed successfully for {self.id}")
                self.status = JobStatus.SUCCEEDED
                # Set the fine-tuned model path
                self.fine_tuned_model = f"{self.id}_model"

                # Log completion event
                self.log(
                    "Training completed successfully",
                    extra_data={"exit_code": 0, "output_dir": output_dir},
                    create_event=True,
                )
            else:
                # Training failed
                logger.error(
                    f"Training failed for {self.id} with exit code {exit_code}"
                )
                self.status = JobStatus.FAILED

                # Log failure event
                self.log(
                    f"Training failed with exit code {exit_code}",
                    level="error",
                    extra_data={"exit_code": exit_code},
                    create_event=True,
                )

        except Exception as e:
            logger.error(f"Error monitoring completion for {self.id}: {e}")
            self.status = JobStatus.FAILED
            self.log(
                f"Completion monitoring error: {e}", level="error", create_event=True
            )
        finally:
            # Always ensure GPU cleanup happens, even if job crashes
            self._ensure_gpu_cleanup()

    def _handle_status_updates(self):
        for status in self.server_comms_handler.receive_status():
            logger.debug(f"Received status update: {status}")

    def cancel(self):
        """Cancel the training job"""
        # Call parent cancel method to check status and set CANCELLED
        super().cancel()

        logger.info(f"Cancelling FileTrainJob {self.id}")

        # Terminate without saving (FileTrainJob doesn't save models mid-training anyway)
        self.terminate(save_model=False)

    def terminate(self, save_model: bool = True):
        """Terminate the training process and clean up resources

        Args:
            save_model: Whether to save model before terminating (not applicable for FileTrainJob)
        """
        logger.info(f"Terminating FileTrainJob {self.id}")

        # Terminate the training process using ProcessRunner
        if self.process_runner:
            self.process_runner.terminate()
            self.process_runner = None

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

        # Release allocated GPUs
        self._ensure_gpu_cleanup()

        logger.info(f"FileTrainJob {self.id} termination completed")

    def _ensure_gpu_cleanup(self):
        """Ensure GPUs are released, even if called multiple times."""
        if self.gpu_manager and self.allocated_gpus:
            try:
                self.gpu_manager.release_gpus(self.id)
                logger.info(
                    f"Released GPUs {self.allocated_gpus} for FileTrainJob {self.id}"
                )
                self.allocated_gpus = None
            except Exception as e:
                logger.error(f"Error releasing GPUs during cleanup: {e}")

    def to_status_model(self) -> JobStatusModel:
        print("To status model", self.model, self.training_file, self.fine_tuned_model)
        return JobStatusModel(
            id=self.id,
            status=self.status.value,
            model=self.model,
            training_file=self.training_file,
            fine_tuned_model=self.fine_tuned_model,
        )
