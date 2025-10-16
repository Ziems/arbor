import json
from trl.data_utils import apply_chat_template

from transformers import AutoTokenizer
from typing import Any, Dict, Sequence
from arbor.server.services.comms.async_batch_requester import ProcessedOutputs
import os
import threading
import time
from datetime import datetime
from typing import Optional, List
from arbor.server.services.comms.async_batch_requester import BatchResult
import coolname

from arbor.server.api.models.schemas import (
    GRPOCheckpointRequest,
    GRPOGPUConfig,
    GRPOInitializeRequest,
    GRPOStatus,
    GRPOStepRequest,
    InferenceJobRequest,
)
from arbor.server.core.config import Config
from arbor.server.services.jobs.inference_job import InferenceJob
from arbor.server.services.jobs.inference_launch_config import InferenceLaunchConfig
from arbor.server.services.jobs.job import Job, JobArtifact
from arbor.server.services.managers.inference_manager import InferenceManager
from arbor.server.services.managers.gpu_manager import GPUManager
from arbor.server.services.scripts.arbor_grpo_config import ArborGRPOConfig
from arbor.server.utils.helpers import get_free_port
from arbor.server.utils.logging import get_logger
from arbor.server.utils.mock_utils import get_script_path, setup_mock_environment
from arbor.server.utils.process_runner import AccelerateProcessRunner
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
        self.gpu_manager: GPUManager = gpu_manager
        self.training_process = None
        self.base_model = None
        self.train_kwargs = None
        self.event_thread = None
        self.saving_checkpoint = False
        self.saving_model = False
        self.terminating = False
        self.training_terminate_pending = False
        self.inference_job: InferenceJob = None
        self.process_runner: Optional[AccelerateProcessRunner] = None
        self.trainer_controller: TrainerControlServer = None
        self.trainer_config: ArborGRPOConfig = None
        self.tokenizer: AutoTokenizer = None

        self.fulfilled_batches: List[BatchResult] = []
        self.pending_batch_ids: List[int] = []
        self.no_submit_streak = 0

        self.checkpoints = {}
        self.last_checkpoint = None
        self.batch_count = 0
        self.last_inference_update = 0
        self.pending_data = set()

    def _make_job_id(self, request: GRPOInitializeRequest):
        slug = coolname.generate_slug(2)
        model = request.model.split("/")[-1].lower()
        name = request.run_name if request.run_name is not None else slug
        timestamp = datetime.now().strftime("%Y%m%d")
        return f"grpo:{model}:{name}:{timestamp}"

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

        wandb_kwargs = {}
        if request.wandb_config is not None:
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
        # Check that the request params are valid
        # Initialize control server client with a self-generated endpoint
        self.trainer_controller = TrainerControlServer()
        self.trainer_controller.start()

        self.tokenizer = AutoTokenizer.from_pretrained(request.model)

        def _allocate_gpus(gpu_config: GRPOGPUConfig):
            if not self.gpu_manager:
                raise RuntimeError("GPU manager is required for GRPO")
            num_inference_gpus = gpu_config.multi.num_inference_gpus
            num_training_gpus = gpu_config.multi.num_training_gpus
            total_gpus = num_inference_gpus + num_training_gpus
            all_gpus = self.gpu_manager.allocate_gpus(self.id, total_gpus)
            inference_gpus = all_gpus[:num_inference_gpus]
            training_gpus = all_gpus[num_inference_gpus:]
            logger.info(
                f"Allocated GPUs {inference_gpus} for inference and {training_gpus} for training"
            )
            return inference_gpus, training_gpus

        inference_gpus, training_gpus = _allocate_gpus(request.gpu_config)

        def _launch_inference_job(
            inference_config: InferenceJobRequest,
            inference_gpus: list[int],
            trainer_controller: TrainerControlServer,
        ):
            # TODO: This "InferenceLaunchConfig"needs to be cleaned up to be more inline with the other config and request structures
            inference_launch_config = InferenceLaunchConfig(
                max_context_length=inference_config.max_context_length,
                gpu_ids=inference_gpus,
                is_grpo=True,
                grpo_job_id=self.id,
            )
            logger.info("Launching inference server...")
            return inference_manager.launch_job(
                request.model, inference_launch_config, self.trainer_controller
            )

        self.inference_job = _launch_inference_job(
            request.inference_config, inference_gpus, self.trainer_controller
        )

        # Set up logging paths for both GRPO and inference jobs
        log_dir = self._make_log_dir()
        self.log_file_path = os.path.join(log_dir, "grpo_training.log")
        if self.inference_job:
            self.inference_job.log_file_path = os.path.join(log_dir, "inference.log")

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

        # Setup mock environment if needed
        my_env = setup_mock_environment(my_env)

        num_processes = len(training_gpus)

        # This is the port for the accelerate main process
        main_process_port = get_free_port()

        logger.info("Running GRPO training command")

        # Use clean process runner for GRPO training
        self.process_runner = AccelerateProcessRunner(self.id)

        self.trainer_config: ArborGRPOConfig = self._build_trainer_config(
            request, log_dir, self.inference_job.port
        )
        # Ensure the trainer binds its control client to our generated endpoint
        self.trainer_config.control_endpoint = self.trainer_controller.endpoint
        self.trainer_config.skip_generation_params_check = True

        config_dict = self.trainer_config.to_dict()
        trainer_config_json = json.dumps(config_dict, separators=(",", ":"))

        # Build script args directly (everything that goes after the script path)
        script_args = [
            # Training args
            "--model",
            request.model,
            "--trainer_config_json",
            trainer_config_json,
            # Comms args
            "--vllm_server_port",
            str(self.inference_job.port),
            "--command_port",
            str(self.trainer_controller.port),
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

        self.trainer_controller.wait_for_clients(num_processes)
        logger.info("Trainer controller clients ready")

        # Start status handling thread
        self.event_thread = threading.Thread(
            target=self._handle_event_updates, args=(), daemon=True
        )
        self.event_thread.start()

    def _handle_submit_batches(self, status: dict):
        pending_batch_ids = status.get("pending_ids", [])
        submitted_any = False
        for batch in self.fulfilled_batches:
            batch_id = batch.batch_id
            if batch_id in pending_batch_ids:
                self.trainer_controller.submit_batch(batch)
                self.fulfilled_batches.remove(batch)
                pending_batch_ids.remove(batch_id)
                submitted_any = True
        self.pending_batch_ids = pending_batch_ids

        if not submitted_any:
            time.sleep(0.5)
            self.no_submit_streak += 1
            if self.no_submit_streak % 10 == 0:
                logger.debug("Waiting for batches to be submitted")
        else:
            self.no_submit_streak = 0

    def _handle_event_updates(self):
        """Handle event updates from training process using ZMQ SUB socket"""
        logger.info("Starting event update handler...")

        try:
            while True:  # TODO: Make this changable with an event set or something
                status = self.trainer_controller.get_status()
                logger.debug(f"Received status: {status}")
                if not status.get("ok", False):
                    logger.error(
                        f"Error getting status: {status.get('error', 'Unknown error')}"
                    )
                    break
                self.wandb_run_id = status.get("wandb_run_id", None)

                self._handle_submit_batches(status)
            # Always ensure GPU cleanup happens, even if job crashes
            self._ensure_gpu_cleanup()
        except Exception as e:
            logger.error(f"Error handling status updates: {e}")

    def validate_batch(self, batch):
        if not isinstance(batch, list):
            raise ValueError("Batch must be a list")

        for item in batch:
            if not isinstance(item, dict):
                raise ValueError("Each item in batch must be a dictionary")
            required_keys = {"messages", "completion", "reward"}
            if not all(key in item for key in required_keys):
                raise ValueError(f"Each item must contain keys: {required_keys}")
        return True

    def grpo_step(self, request: GRPOStepRequest) -> str:
        self.validate_batch(request.batch)

        def _handle_group_data(group: list[dict]):
            batch_result = build_batch_result(
                batch_id=self.batch_count,
                tokenizer=self.tokenizer,
                samples=group,
                max_prompt_length=self.trainer_config.max_prompt_length,
                max_seq_len=self.trainer_config.max_seq_len,
                num_generations=self.trainer_config.num_generations,
                trainer_config=self.trainer_config,
            )

            self.trainer_controller.submit_batch(batch_result)

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
        self.saving_checkpoint = True
        try:
            response = self.trainer_controller.request_checkpoint(
                request.checkpoint_name
            )
        finally:
            self.saving_checkpoint = False

        checkpoint_dir = response.get("checkpoint_dir") if response else None
        if checkpoint_dir:
            self.checkpoints[request.checkpoint_name] = checkpoint_dir
            self.last_checkpoint = request.checkpoint_name
            logger.info(
                f"Checkpoint '{request.checkpoint_name}' saved at {checkpoint_dir}"
            )
        else:
            logger.warning(
                f"Checkpoint response missing directory for {request.checkpoint_name}"
            )

        return checkpoint_dir

    def terminate_training(self, timeout: float = 300.0):
        if not self.trainer_controller:
            raise RuntimeError("Trainer controller is not initialized for this job")

        if self.training_terminate_pending:
            logger.info("Terminate request already in progress for this job")
            return

        self.training_terminate_pending = True
        try:
            logger.info("Sending terminate request to trainer")
            response = self.trainer_controller.request_terminate()
            logger.debug(f"Trainer terminate response: {response}")

            if self.process_runner and self.process_runner.is_running():
                logger.info("Waiting for training process to exit after terminate request")
                deadline = time.time() + timeout
                while time.time() < deadline and self.process_runner.is_running():
                    time.sleep(5)
                    logger.info("Training process still terminating...")

                if self.process_runner.is_running():
                    logger.warning(
                        "Training process still running after terminate timeout; proceeding without force termination"
                    )
            else:
                logger.info("Training process already stopped")
            return response
        finally:
            self.training_terminate_pending = False

    def cancel(self):
        """Cancel the GRPO training job"""
        # Call parent cancel method to check status and set CANCELLED
        super().cancel()

        logger.info(f"Cancelling GRPOJob {self.id}")

        # Terminate without saving model for faster cancellation
        self.terminate_process()


    def terminate_process(self):
        try:
            # Terminate training process using ProcessRunner
            if self.process_runner:
                logger.info("Terminating training process...")
                self.process_runner.terminate()
                self.process_runner = None

            if self.inference_job and self.inference_job.process is not None:
                logger.info("Terminating inference job...")
                self.inference_job.terminate()

            # Release GPUs
            self._ensure_gpu_cleanup()

            # Reinitialize in case we want to start a new training run
            self.training_process = None
            self.process_runner = None
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
            pending_batch_ids=self.pending_batch_ids,
        )

    def _handle_checkpoint_saved_event(self, event: dict):
        logger.info("Received checkpoint saved status")
        self.checkpoints[event["checkpoint_name"]] = event["output_dir"]
        self.last_checkpoint = event["checkpoint_name"]
        self.saving_checkpoint = False
        logger.info("Checkpoint saved")

    def _handle_error_event(self, event: dict):
        error_msg = event.get("error", "Unknown error")
        logger.error(f"Training error: {error_msg}")

    def _handle_terminated_event(self, event: dict):
        self.terminating = False
        logger.info("Training process terminated")


def build_batch_result(
    batch_id: int,
    tokenizer: AutoTokenizer,
    samples: Sequence[Dict[str, Any]],
    max_prompt_length: int | None,
    max_seq_len: int,
    num_generations: int,
    trainer_config: ArborGRPOConfig,
):
    if not samples:
        raise ValueError("No samples provided to build batch result")

    if len(samples) != num_generations:
        raise ValueError(
            f"Expected {num_generations} samples in the group, received {len(samples)}"
        )

    effective_max_prompt = max_prompt_length or max_seq_len

    prompt_ids: List[List[int]] = []
    prompt_mask: List[List[int]] = []
    completion_ids: List[List[int]] = []
    completion_mask: List[List[int]] = []
    rewards: List[float] = []

    prompt_completion_texts = [
        apply_chat_template(
            {
                "prompt": sample["messages"],
                "completion": (
                    sample["completion"]
                    if isinstance(sample["completion"], list)
                    else [sample["completion"]]
                ),
            },
            tokenizer,
        )
        for sample in samples
    ]

    prompts_text = [
        prompt_completion_text["prompt"]
        for prompt_completion_text in prompt_completion_texts
    ]
    prompt_inputs = tokenizer(
        prompts_text,
        padding=True,
        padding_side="left",
        add_special_tokens=False,
    )
    prompt_ids, prompt_mask = (
        prompt_inputs["input_ids"],
        prompt_inputs["attention_mask"],
    )

    completions_text = [
        prompt_completion_text["completion"]
        for prompt_completion_text in prompt_completion_texts
    ]
    completion_inputs = tokenizer(
        completions_text,
        padding=True,
        add_special_tokens=False,
    )
    completion_ids, completion_mask = (
        completion_inputs["input_ids"],
        completion_inputs["attention_mask"],
    )

    rewards = [float(sample["reward"]) for sample in samples]

    for prompt_id, prompt_mask, completion_id, completion_mask, reward in zip(
        prompt_ids, prompt_mask, completion_ids, completion_mask, rewards
    ):
        if len(prompt_id) > effective_max_prompt:
            logger.warning(
                f"Prompt length {len(prompt_id)} is greater than effective max prompt length {effective_max_prompt}"
            )

        if (
            trainer_config.max_completion_length
            and len(completion_id) > trainer_config.max_completion_length
        ):
            logger.warning(
                f"Completion length {len(completion_id)} is greater than effective max completion length {trainer_config.max_completion_length}"
            )

        # TODO: This should exist
        # if trainer_config.max_model_length and len(prompt_id + completion_id) > trainer_config.max_model_length:
        #     logger.warning(f"Prompt + completion length {len(prompt_id + completion_id)} is greater than effective max model length {trainer_config.max_model_length}")

    processed_results = ProcessedOutputs(
        prompt_ids=prompt_inputs["input_ids"],
        prompt_mask=prompt_inputs["attention_mask"],
        completion_ids=completion_inputs["input_ids"],
        completion_mask=completion_inputs["attention_mask"],
        rewards=rewards,
    )

    return BatchResult(
        batch_id=batch_id,
        prompts=prompts_text,
        completions=completions_text,
        processed_results=processed_results,
        all_reward_dict={"reward": rewards},
    )
