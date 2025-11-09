import json
import os
import threading
import time
from datetime import datetime
from typing import Any, Optional, Sequence, cast

import coolname
from transformers import AutoTokenizer
from trl.data_utils import apply_chat_template

from arbor.core.config import Config
from arbor.core.logging import get_logger
from arbor.server.api.schemas import (
    GRPOCheckpointRequest,
    GRPOGPUConfig,
    GRPOInitializeRequest,
    GRPOStatus,
    GRPOStepRequest,
    InferenceLaunchOwner,
    InferenceLaunchRequest,
    JobStatus,
)
from arbor.server.services.comms.async_batch_requester import (
    BatchResult,
    ProcessedOutputs,
)
from arbor.server.services.comms.control_server import TrainerControlServer
from arbor.server.services.jobs.inference_job import InferenceJob
from arbor.server.services.jobs.job import Job, JobArtifact
from arbor.server.services.managers.gpu_manager import GPUManager
from arbor.server.services.managers.inference_manager import InferenceManager
from arbor.training.grpo.config import ArborGRPOConfig
from arbor.utils.helpers import get_free_port
from arbor.utils.process_runner import AccelerateProcessRunner

logger = get_logger(__name__)


class GRPOJob(Job):
    def __init__(
        self, config: Config, request: GRPOInitializeRequest, gpu_manager=None
    ):
        id = self._make_job_id(request)
        # GRPO jobs need logs, models, and metrics
        super().__init__(
            config,
            id=id,
            artifacts=[JobArtifact.LOGS, JobArtifact.CHECKPOINTS, JobArtifact.METRICS],
        )
        self.gpu_manager: GPUManager = gpu_manager
        self.training_process = None
        self.event_thread = None
        self.training_terminate_pending = False
        self.inference_job: Optional[InferenceJob] = None
        self.process_runner: Optional[AccelerateProcessRunner] = None
        self.trainer_controller: TrainerControlServer = None
        self.trainer_config: ArborGRPOConfig = None
        self.tokenizer: AutoTokenizer = None

        self.fulfilled_batches: list[BatchResult] = []
        self.pending_batch_ids: list[int] = []
        self.no_submit_streak = 0

        self.batch_count = 0
        self.last_inference_update = 0
        self.pending_data = set()

        self.checkpoints: dict[str, dict[str, Any]] = {}
        self.last_checkpoint: str | None = None

    def _update_checkpoint_records(self, records: list[dict[str, Any]] | None) -> None:
        if not records:
            return

        checkpoints = {record["checkpoint_name"]: record for record in records}
        self.checkpoints = checkpoints
        latest = max(records, key=lambda r: r.get("timestamp", 0))
        self.last_checkpoint = latest.get("checkpoint_name")

    def _make_job_id(self, request: GRPOInitializeRequest):
        slug = coolname.generate_slug(2)
        model = request.model.split("/")[-1].lower()
        name = request.run_name if request.run_name is not None else slug
        timestamp = datetime.now().strftime("%Y%m%d")
        return f"grpo:{model}:{name}:{timestamp}"

    def _build_trainer_config(
        self, request: GRPOInitializeRequest, output_dir: str, vllm_port: int
    ) -> ArborGRPOConfig:
        if output_dir is None:
            raise ValueError("output_dir is required to build ArborGRPOConfig")

        trainer_kwargs = request.trainer_config.model_dump(
            exclude_unset=True,
        )

        config = ArborGRPOConfig(**trainer_kwargs)

        config.output_dir = output_dir
        config.vllm_server_port = vllm_port
        config.run_name = self.id
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
            inference_config: InferenceLaunchRequest,
            inference_gpus: list[int],
            trainer_controller: TrainerControlServer,
            log_file_path: str,
        ):
            launch_request = inference_config.model_copy(
                update={
                    "gpu_ids": inference_gpus,
                    "owner": InferenceLaunchOwner(type="grpo", job_id=self.id),
                    "log_file_path": log_file_path,
                }
            )
            logger.info("Launching inference server...")
            return inference_manager.launch_from_request(
                launch_request,
                trainer_controller=trainer_controller,
                reuse_existing=False,
                preallocated_gpu_ids=inference_gpus,
            )

        log_dir = self._make_log_dir()
        checkpoint_dir = self._make_checkpoints_dir()
        self.log_file_path = os.path.join(log_dir, "grpo_training.log")
        inference_log_path = os.path.join(log_dir, "inference.log")

        self.inference_job = _launch_inference_job(
            request.inference_config,
            inference_gpus,
            self.trainer_controller,
            inference_log_path,
        )

        trainer_module = "arbor.training.grpo.trainer"

        my_env = os.environ.copy()
        # Use the training GPUs that were allocated earlier
        gpu_ids_str = ",".join(map(str, training_gpus))

        my_env["CUDA_VISIBLE_DEVICES"] = gpu_ids_str

        my_env["WANDB_SILENT"] = "true"

        num_processes = len(training_gpus)

        # This is the port for the accelerate main process
        main_process_port = get_free_port()

        logger.info("Running GRPO training command")

        # Use clean process runner for GRPO training
        self.process_runner = AccelerateProcessRunner(self.id)

        self.trainer_config: ArborGRPOConfig = self._build_trainer_config(
            request, checkpoint_dir, self.inference_job.port
        )
        # Ensure the trainer binds its control client to our generated endpoint
        self.trainer_config.control_endpoint = self.trainer_controller.endpoint
        self.trainer_config.steps_per_generation = None

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
            module=trainer_module,
            num_processes=num_processes,
            main_process_port=main_process_port,
            script_args=script_args,
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
        self.status = JobStatus.RUNNING

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
                self._update_checkpoint_records(status.get("checkpoints"))

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

    def grpo_step(self, request: GRPOStepRequest):
        self.validate_batch(request.batch)

        def _handle_group_data(group: list[dict[str, Any]]):
            batch_result = build_batch_result(
                batch_id=self.batch_count,
                tokenizer=self.tokenizer,
                samples=group,
                max_prompt_length=self.trainer_config.max_prompt_length,
                max_seq_len=self.trainer_config.max_seq_len,
                num_generations=self.trainer_config.num_generations,
                trainer_config=self.trainer_config,
                metrics=request.metrics if hasattr(request, "metrics") else None,
            )

            self.trainer_controller.submit_batch(batch_result)

            self.batch_count += 1

        try:
            if isinstance(request.batch[0], list):
                # Handle List[List[dict]] case
                groups = cast(list[list[dict[str, Any]]], request.batch)
                for group in groups:
                    _handle_group_data(group)
            else:
                # Handle List[dict] case
                group = cast(list[dict[str, Any]], request.batch)
                _handle_group_data(group)

        except Exception as e:
            logger.error(f"Failed to send batch to training process: {e}")
            raise

    def checkpoint(self, request: GRPOCheckpointRequest) -> GRPOStatus:
        if not self.trainer_controller:
            raise RuntimeError("Trainer controller is not initialized for this job")

        response = self.trainer_controller.request_checkpoint(
            request.checkpoint_name, request.metadata
        )
        if not response.get("ok", False):
            raise RuntimeError(
                f"Checkpoint request failed: {response.get('error', 'Unknown error')}"
            )
        checkpoint = response.get("checkpoint")
        if not checkpoint:
            raise RuntimeError("Trainer did not return checkpoint metadata")

        self.checkpoints[checkpoint["checkpoint_name"]] = checkpoint
        self.last_checkpoint = checkpoint["checkpoint_name"]
        logger.info(f"Checkpoint completed for {checkpoint['checkpoint_name']}")
        return self.get_status()

    def save_final_checkpoint(self, checkpoint_name: str = "checkpoint_final") -> None:
        """Attempt to save a final checkpoint before shutting down training."""
        if not self.trainer_controller:
            logger.warning(
                "Skipping final checkpoint because trainer controller is not initialized"
            )
            return

        try:
            request = GRPOCheckpointRequest(
                job_id=self.id, checkpoint_name=checkpoint_name
            )
            self.checkpoint(request)
        except Exception as exc:  # pragma: no cover - best effort logging
            logger.error(f"Failed to save final checkpoint: {exc}")

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
                logger.info("Terminating training process after terminate request")
                self.process_runner.terminate()
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

    def terminate_process(
        self,
        *,
        stop_inference: bool = True,
        release_gpus: bool = True,
    ) -> None:
        training_log_path = self.log_file_path
        inference_log_path = (
            getattr(self.inference_job, "log_file_path", None)
            if self.inference_job is not None
            else None
        )
        try:
            # Terminate training process using ProcessRunner
            if self.process_runner:
                logger.info("Terminating training process...")
                self.process_runner.terminate()
                self.process_runner = None

            if (
                stop_inference
                and self.inference_job
                and self.inference_job.process is not None
            ):
                logger.info("Terminating inference job...")
                self.inference_job.terminate()
                self.inference_job = None
            elif not stop_inference:
                logger.info(
                    "Leaving inference server running after training termination"
                )
                if self.inference_job:
                    self.inference_job.promote_to_standalone()

            if release_gpus:
                # Release GPUs when we're fully shutting down the job
                self._ensure_gpu_cleanup()
            else:
                logger.info("Skipping GPU cleanup to keep inference resources reserved")

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
            if stop_inference:
                self.inference_job = None
            self.server_comms_handler = None
            self.event_thread = None
            self.batch_count = 0
        finally:
            if training_log_path:
                logger.info("Training log saved to %s", training_log_path)
            if inference_log_path:
                logger.info("Inference log saved to %s", inference_log_path)

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


def build_batch_result(
    batch_id: int,
    tokenizer: AutoTokenizer,
    samples: Sequence[dict[str, Any]],
    max_prompt_length: int | None,
    max_seq_len: int | None,
    num_generations: int,
    trainer_config: ArborGRPOConfig,
    metrics: dict[str, Any] | None = None,
):
    if not samples:
        raise ValueError("No samples provided to build batch result")

    if len(samples) != num_generations:
        raise ValueError(
            f"Expected {num_generations} samples in the group, received {len(samples)}"
        )

    effective_max_prompt = max_prompt_length or max_seq_len

    prompt_ids: list[list[int]] = []
    prompt_mask: list[list[int]] = []
    completion_ids: list[list[int]] = []
    completion_mask: list[list[int]] = []
    rewards: list[float] = []

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
    # calculate completion lengths
    completion_lengths = [sum(cm) for cm in completion_mask]

    rewards = [float(sample["reward"]) for sample in samples]
    if trainer_config.soft_completion_penalty_length is not None:
        processed_rewards = []
        for (
            _prompt_id,
            _prompt_mask,
            _completion_id,
            _completion_mask,
            _reward,
            _completion_length,
        ) in zip(
            prompt_ids,
            prompt_mask,
            completion_ids,
            completion_mask,
            rewards,
            completion_lengths,
        ):
            if _completion_length > trainer_config.soft_completion_penalty_length:
                if _reward > 0:
                    new_reward = _reward * (
                        1
                        - (
                            _completion_length
                            - trainer_config.soft_completion_penalty_length
                        )
                        / (
                            trainer_config.max_completion_length  # type: ignore
                            - trainer_config.soft_completion_penalty_length
                        )
                    )
                else:
                    new_reward = _reward
                logger.info(
                    f"Applying soft completion penalty to completion {_completion_length} with reward {_reward} -> {new_reward}"
                )
                _reward = new_reward
            processed_rewards.append(_reward)
        rewards = processed_rewards

    for prompt_id, _prompt_mask, completion_id, _completion_mask, _reward in zip(
        prompt_ids, prompt_mask, completion_ids, completion_mask, rewards
    ):
        if effective_max_prompt is not None and len(prompt_id) > effective_max_prompt:
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
        metrics=metrics or {},
    )
