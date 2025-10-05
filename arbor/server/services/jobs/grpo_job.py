import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import coolname

from arbor.server.api.models.schemas import (
    GRPOCheckpointRequest,
    GRPOGPUConfig,
    GRPOInitializeRequest,
    GRPOStatus,
    GRPOStepRequest,
)
from arbor.server.core.config import Config
from arbor.server.services.jobs.inference_job import InferenceJob
from arbor.server.services.jobs.inference_launch_config import InferenceLaunchConfig
from arbor.server.services.jobs.job import Job, JobArtifact
from arbor.server.services.managers.inference_manager import InferenceManager
from arbor.server.services.scripts.arbor_grpo_config import ArborGRPOConfig
from arbor.server.utils.helpers import get_free_port
from arbor.server.utils.logging import get_logger
from arbor.server.utils.mock_utils import get_script_path, setup_mock_environment
from arbor.server.utils.process_runner import AccelerateProcessRunner
from arbor.server.services.api.models.schemas import InferenceJobRequest
from arbor.server.services.comms.control_server import TrainerControlServer

logger = get_logger(__name__)


class GRPOJob(Job):
    def __init__(
        self, config: Config, request: GRPOInitializeRequest, gpu_manager=None
    ):
        id = self._make_job_id(request)
        # GRPO jobs need all artifact types - logs, models, checkpoints, and metrics
        super().__init__(
            config,
            id=id,
            artifacts=[
                JobArtifact.LOGS,
                JobArtifact.MODEL,
                JobArtifact.CHECKPOINTS,
                JobArtifact.METRICS,
            ],
        )
        self.gpu_manager = gpu_manager
        self.training_process = None
        self.base_model = None
        self.train_kwargs = None
        self.server_comms_handler = None
        self.event_thread = None
        self.saving_checkpoint = False
        self.saving_model = False
        self.terminating = False
        self.inference_job: InferenceJob = None
        self.process_runner: Optional[AccelerateProcessRunner] = None

        self.checkpoints = {}
        self.last_checkpoint = None
        self.batch_count = 0
        self.last_inference_update = 0
        self.pending_data = set()

    def _make_job_id(self, request: GRPOInitializeRequest):
        slug = coolname.generate_slug(2)
        model = request.model.split("/")[-1].lower()
        suffix = request.suffix if request.suffix is not None else slug
        timestamp = datetime.now().strftime("%Y%m%d")
        return f"grpo:{model}:{suffix}:{timestamp}"

    def find_training_args(self, request: GRPOInitializeRequest) -> dict:
        """Process the config request and return training arguments."""
        output_dir = self._make_model_dir()  # Use base class method

        # Here are defaults for training. We can adjust them if we disagree w the huggingface defaults
        default_train_kwargs = {"output_dir": output_dir, "grpo_flavor": "grpo"}

        train_kwargs = request.model_dump(exclude_unset=True)
        return {**default_train_kwargs, **(train_kwargs or {})}


    def _build_trainer_config(
        self, request: GRPOInitializeRequest, output_dir: str, vllm_port: int
    ) -> ArborGRPOConfig:
        if output_dir is None:
            raise ValueError("output_dir is required to build ArborGRPOConfig")
        
        trainer_kwargs = request.trainer_config.model_dump(
            exclude_unset=True,
        )
        wandb_kwargs = request.wandb_config.model_dump(
            exclude_unset=True,
        )

        config = ArborGRPOConfig(**trainer_kwargs, **wandb_kwargs)

        config.output_dir = output_dir
        config.vllm_server_port = vllm_port
        return config


    def initialize(
        self, request: GRPOInitializeRequest, inference_manager: InferenceManager
    ):
        def _allocate_gpus(gpu_config: GRPOGPUConfig):
            if not self.gpu_manager:
                raise RuntimeError("GPU manager is required for GRPO")
            num_inference_gpus = gpu_config.multi.num_inference_gpus
            num_training_gpus = gpu_config.multi.num_training_gpus
            total_gpus = num_inference_gpus + num_training_gpus
            all_gpus = self.gpu_manager.allocate_gpus(self.id, total_gpus)
            inference_gpus = all_gpus[:num_inference_gpus]
            training_gpus = all_gpus[num_inference_gpus:]
            self.logger.info(
                f"Allocated GPUs {inference_gpus} for inference and {training_gpus} for training"
            )
            return inference_gpus, training_gpus

        inference_gpus, training_gpus = _allocate_gpus(request.gpu_config)

        def _launch_inference_job(inference_config: InferenceJobRequest, inference_gpus: list[int]):
            # TODO: This "InferenceLaunchConfig"needs to be cleaned up to be more inline with the other config and request structures
            inference_launch_config = InferenceLaunchConfig(
                max_context_length=inference_config.max_context_length,
                gpu_ids=inference_gpus,
                is_grpo=True,
                grpo_job_id=self.id,
            )
            logger.info("Launching inference server...")
            return inference_manager.launch_job(request.model, inference_launch_config)

        self.inference_job = _launch_inference_job(request.inference_config, inference_gpus)


        # Set up logging paths for both GRPO and inference jobs
        log_dir = self._make_log_dir()
        self.log_file_path = os.path.join(log_dir, "grpo_training.log")
        if self.inference_job:
            self.inference_job.log_file_path = os.path.join(log_dir, "inference.log")

        # Initialize ZMQ socket manager - no need for connection acceptance thread anymore
        self.trainer_controller = TrainerControlServer(
            endpoint=f"tcp://{self.server_comms_handler.host}:{self.server_comms_handler.command_port}"
        )

        script_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"
        )
        script_name = "arbor_grpo_trainer.py"
        script_path = get_script_path(script_name, script_dir)

        my_env = os.environ.copy()
        # Use the training GPUs that were allocated earlier
        gpu_ids_str = ",".join(map(str, training_gpus))

        my_env["CUDA_VISIBLE_DEVICES"] = gpu_ids_str

        # Handle WandB configuration
        if request.wandb_config is not None:
            # WandB is explicitly requested, just silence login prompts
            my_env["WANDB_SILENT"] = "true"
        else:
            # WandB not requested, disable it completely to avoid login errors
            my_env["WANDB_SILENT"] = "true"
            request.wandb_config.project = "none"

        # Setup mock environment if needed
        my_env = setup_mock_environment(my_env)

        num_processes = len(training_gpus)

        # This is the port for the accelerate main process
        main_process_port = get_free_port()

        logger.info(f"Running GRPO training command")

        # Use clean process runner for GRPO training
        self.process_runner = AccelerateProcessRunner(self.id)

        trainer_config: ArborGRPOConfig = self._build_trainer_config(
            request, self.inference_job.port
        )
        trainer_config_json = json.dumps(
            trainer_config.to_sanitized_dict(), separators=(",", ":")
        )

        # Build script args directly (everything that goes after the script path)
        script_args = [
            # Training args
            "--model", request.model,
            "--trainer_config_json", trainer_config_json,
             # Comms args
            "--vllm_server_port", str(self.inference_job.port),
            "--command_port", str(self.server_comms_handler.command_port),
       ]

        self.training_process = self.process_runner.start_training(
            script_path=script_path,
            num_processes=num_processes,
            main_process_port=main_process_port,
            script_args=script_args,
            accelerate_config=self.config.accelerate_config,
            env=my_env,
            log_callback=self.create_log_callback("GRPO"),
        )

        # Start status handling thread
        self.event_thread = threading.Thread(
            target=self._handle_event_updates, args=(), daemon=True
        )
        self.event_thread.start()
        self.server_comms_handler.wait_for_clients(num_processes)

    def _handle_event_updates(self):
        """Handle event updates from training process using ZMQ SUB socket"""
        logger.info("Starting event update handler...")
        try:
            for event in self.server_comms_handler.receive_event():
                event_type = event.get("type")
                logger.debug(f"Received event: {event}")
                if event_type == "weight_update_request":
                    self._handle_weight_update_request_event(event)

                elif event_type == "weight_update_complete":
                    # Training has completed the weight update
                    self._handle_weight_update_complete_event(event)

                elif event_type == "checkpoint_saved":
                    self._handle_checkpoint_saved_event(event)

                elif event_type == "data_processed":
                    self._handle_data_processed_event(event)

                elif event_type == "error":
                    self._handle_error_event(event)

                elif event_type == "terminated":
                    self._handle_terminated_event(event)
                else:
                    logger.warning(f"Received unknown event: {event}")
            # Make sure to allow inference if there's an error
            try:
                self.inference_job.complete_weight_update()
            except:
                pass

            # Always ensure GPU cleanup happens, even if job crashes
            self._ensure_gpu_cleanup()
        except Exception as e:
            logger.error(f"Error handling status updates: {e}")

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

        def _handle_group_data(group: list[dict]):
            # Add metadata to each item in the group
            for idx, item in enumerate(group):
                if isinstance(item, dict):
                    item_id = f"{self.batch_count}"
                    item["_metadata"] = {
                        "batch_id": item_id,
                        "timestamp": time.time(),
                    }
                    self.pending_data.add(item_id)

            self.server_comms_handler.send_data(group)
            self.batch_count += 1

        try:
            if isinstance(request.batch[0], list):
                # Handle List[List[dict]] case
                for group in request.batch:
                    _handle_group_data(group)
            else:
                # Handle List[dict] case
                _handle_group_data(request.batch)

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

    def cancel(self):
        """Cancel the GRPO training job"""
        # Call parent cancel method to check status and set CANCELLED
        super().cancel()

        logger.info(f"Cancelling GRPOJob {self.id}")

        # Terminate without saving model for faster cancellation
        self.terminate(save_model=False)

    def terminate(self, save_model: bool = True):
        """Clean up resources and optionally save the final model.

        Args:
            save_model: Whether to save the model before terminating
        """
        # if save_model:
        #     logger.info("Terminating with model saving...")
        #     time.sleep(5)

        #     while (
        #         self.inference_job and self.inference_job.is_updating
        #     ):  # Use the property instead of direct access
        #         logger.info(
        #             "Waiting for final weight updates to finish before saving..."
        #         )
        #         time.sleep(5)

        #     logger.info("Sending save model command")
        #     self.saving_model = True
        #     self.server_comms_handler.send_command({"command": "save_model"})
        #     while self.saving_model:
        #         logger.info("Waiting for final model to be saved...")
        #         time.sleep(5)
        # else:
        #     logger.info("Terminating without model saving...")

        # Send termination command if we have comms
        if self.server_comms_handler:
            try:
                logger.info("Sending termination command")
                self.terminating = True
                self.server_comms_handler.send_command({"command": "terminate"})

                # Wait time depends on whether we're saving model or not
                max_wait_time = 15 if save_model else 5
                start_time = time.time()
                while self.terminating:
                    if time.time() - start_time > max_wait_time:
                        logger.warning(
                            f"Termination wait timed out after {max_wait_time} seconds, proceeding with cleanup..."
                        )
                        break
                    logger.info("Waiting for run to be terminated...")
                    time.sleep(3)
            except Exception as e:
                logger.warning(f"Error sending termination command: {e}")

        logger.info("Starting cleanup")
        self.cleanup_termination()

        if save_model and self.train_kwargs and "output_dir" in self.train_kwargs:
            # output_dir = self.train_kwargs["output_dir"]
            # logger.info(f"Training completed. Model saved to {output_dir}")
            # logger.info(f"Training logs and checkpoints are stored in: {output_dir}")
            # if not os.path.exists(output_dir):
            #     logger.warning(f"Output directory {output_dir} does not exist")
            self.train_kwargs = None
        else:
            logger.info(
                "Training terminated, no output directory specified"
                + (" (model not saved)" if not save_model else "")
            )
            self.train_kwargs = None

    def cleanup_termination(self):
        try:
            # Terminate training process using ProcessRunner
            if self.process_runner:
                logger.info("Terminating training process...")
                self.process_runner.terminate()
                self.process_runner = None

            # Clean up ZMQ connections
            if self.server_comms_handler:
                logger.debug("Closing ZMQ connections...")
                self.server_comms_handler.close()

            if self.inference_job and self.inference_job.process is not None:
                logger.info("Terminating inference job...")
                self.inference_job.terminate()

            # Release GPUs
            self._ensure_gpu_cleanup()

            # Reinitialize in case we want to start a new training run
            self.training_process = None
            self.process_runner = None
            self.server_comms_handler = None
            self.event_thread = None
            self.batch_count = 0
            logger.info("Cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            # Still reset state even if cleanup fails
            self.training_process = None
            self.process_runner = None
            self.server_comms_handler = None
            self.event_thread = None
            self.batch_count = 0

    def _ensure_gpu_cleanup(self):
        """Ensure GPUs are released, even if called multiple times."""
        if self.gpu_manager:
            try:
                self.gpu_manager.release_gpus(self.id)
                logger.info(f"Released GPUs for GRPO job {self.id}")
            except Exception as e:
                logger.error(f"Error releasing GPUs during cleanup: {e}")

    def get_status(self) -> GRPOStatus:
        return GRPOStatus(
            job_id=self.id,
            status=self.status.value,
            current_model=self.id,
            checkpoints=self.checkpoints,
            last_checkpoint=self.last_checkpoint,
        )

    def _handle_weight_update_request_event(self, event: dict):
        # Training is requesting to start a weight update
        logger.debug("Received weight update request from training...")
        logger.debug("Blocking new inference calls...")
        self.inference_job.start_weight_update()

        # Wait for all existing inference requests to complete
        logger.debug("Waiting for existing inference requests to complete...")
        max_wait_time = 30  # Maximum time to wait for existing requests
        wait_start = time.time()
        last_log_time = wait_start

        while self.inference_job.has_active_requests:
            active_count = self.inference_job.active_request_count

            # Only log every 10 seconds to reduce spam
            current_time = time.time()
            if current_time - last_log_time >= 10:
                logger.info(
                    f"Waiting for {active_count} active inference requests to complete..."
                )
                last_log_time = current_time

            if current_time - wait_start > max_wait_time:
                logger.warning(
                    f"Timeout waiting for inference requests to complete after {max_wait_time}s, proceeding anyway..."
                )
                break

            time.sleep(0.5)  # Check every 500ms

        logger.info(
            "All inference requests completed, sending ready signal to training..."
        )
        self.server_comms_handler.send_command({"command": "weight_update_ready"})

    def _handle_weight_update_complete_event(self, event: dict):
        logger.debug("Weight update completed, allowing new inference calls...")
        self.inference_job.complete_weight_update()

    def _handle_checkpoint_saved_event(self, event: dict):
        logger.info("Received checkpoint saved status")
        self.checkpoints[event["checkpoint_name"]] = event["output_dir"]
        self.last_checkpoint = event["checkpoint_name"]
        self.saving_checkpoint = False
        logger.info("Checkpoint saved")

    def _handle_data_processed_event(self, event: dict):
        # Extract the batch_id from the processed data
        processed_data = event.get("processed_data")
        if processed_data and isinstance(processed_data, dict):
            batch_id = processed_data.get("_metadata", {}).get("batch_id")
            if batch_id is not None:
                # Remove the batch_id from pending set
                if batch_id in self.pending_data:
                    self.pending_data.remove(batch_id)
                    logger.debug(f"Removed batch_id {batch_id} from pending set")
                else:
                    logger.warning(f"batch_id {batch_id} not found in pending set")
            else:
                logger.warning("Processed data missing metadata batch_id")
        else:
            logger.warning("Data processed event missing processed_data")

    def _handle_error_event(self, event: dict):
        error_msg = event.get("error", "Unknown error")
        logger.error(f"Training error: {error_msg}")

    def _handle_terminated_event(self, event: dict):
        self.terminating = False
        logger.info("Training process terminated")
