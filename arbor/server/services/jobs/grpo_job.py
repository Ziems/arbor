import json

from transformers import AutoTokenizer
import copy
from typing import Any, Dict, Sequence
import copy
from arbor.server.services.comms.async_batch_requester import ProcessedOutputs
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Iterable
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
        self.gpu_manager = gpu_manager
        self.training_process = None
        self.base_model = None
        self.train_kwargs = None
        self.event_thread = None
        self.saving_checkpoint = False
        self.saving_model = False
        self.terminating = False
        self.inference_job: InferenceJob = None
        self.process_runner: Optional[AccelerateProcessRunner] = None
        self.trainer_controller: TrainerControlServer = None
        self.trainer_config: ArborGRPOConfig = None
        self.tokenizer: Any = None

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
        # Initialize control server client with a self-generated endpoint
        self.trainer_controller = TrainerControlServer()
        self.trainer_controller.start()

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

        def _launch_inference_job(inference_config: InferenceJobRequest, inference_gpus: list[int], trainer_controller: TrainerControlServer):
            # TODO: This "InferenceLaunchConfig"needs to be cleaned up to be more inline with the other config and request structures
            inference_launch_config = InferenceLaunchConfig(
                max_context_length=inference_config.max_context_length,
                gpu_ids=inference_gpus,
                is_grpo=True,
                grpo_job_id=self.id,
            )
            logger.info("Launching inference server...")
            return inference_manager.launch_job(request.model, inference_launch_config, self.trainer_controller)

        self.inference_job = _launch_inference_job(request.inference_config, inference_gpus, self.trainer_controller)

        self.tokenizer = AutoTokenizer.from_pretrained(request.model)

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

        logger.info(f"Running GRPO training command")

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
            "--model", request.model,
            "--trainer_config_json", trainer_config_json,
             # Comms args
            "--vllm_server_port", str(self.inference_job.port),
            "--command_port", str(self.trainer_controller.port),
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
            while True: # TODO: Make this changable with an event set or something
                status = self.trainer_controller.get_status()
                logger.debug(f"Received status: {status}")
                if not status.get("ok", False):
                    logger.error(f"Error getting status: {status.get('error', 'Unknown error')}")
                    break

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
        while self.saving_checkpoint:
            logger.info(
                "Saving checkpoint, pausing GRPO steps until checkpoint is saved..."
            )
            time.sleep(5)

        self.validate_batch(request.batch)

        def _handle_group_data(group: list[dict]):

            batch_result = build_batch_result_from_samples(
                batch_id=self.batch_count,
                tokenizer=self.tokenizer,
                samples=group,
                max_prompt_length=self.trainer_config.max_prompt_length,
                max_seq_len=self.trainer_config.max_seq_len,
                num_generations=self.trainer_config.num_generations,
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
        # while (
        #     self.inference_job.is_updating
        # ):  # Use the property instead of direct access
        #     logger.info("Waiting for weight updates to finish before checkpointing...")
        #     time.sleep(5)

        # self.saving_checkpoint = True
        # self.trainer_controller.request_checkpoint()
        # while self.saving_checkpoint:
        #     logger.info("Waiting for checkpoint to be saved...")
        #     time.sleep(5)
        pass

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

def _tokenize(tokenizer: Any, text: str) -> List[int]:
    if hasattr(tokenizer, "encode"):
        return tokenizer.encode(text, add_special_tokens=False)
    tokens = tokenizer(  # type: ignore[operator]
        text,
        add_special_tokens=False,
        return_attention_mask=False,
    )
    if isinstance(tokens, dict):
        return list(tokens.get("input_ids", []))
    return list(tokens)


def _render_messages(messages: Any) -> str:
    if not isinstance(messages, (list, tuple)):
        return str(messages or "")
    lines: List[str] = []
    for message in messages:
        if not isinstance(message, dict):
            lines.append(str(message))
            continue
        role = message.get("role", "user")
        content = message.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _normalize_prompt(obj: Any) -> str:
    if isinstance(obj, dict) and "messages" in obj:
        return _render_messages(obj["messages"])
    return str(obj or "")


def _normalize_completion(obj: Any) -> str:
    if isinstance(obj, dict) and "messages" in obj:
        messages = obj["messages"]
        if messages:
            return str(messages[-1].get("content", ""))
        return ""
    return str(obj or "")


def build_batch_result_from_samples(
    *,
    batch_id: int,
    tokenizer: Any,
    samples: Sequence[Dict[str, Any]],
    max_prompt_length: int | None,
    max_seq_len: int,
    num_generations: int,
) -> BatchResult:
    """Treat the provided samples as a single generation group."""

    if not samples:
        raise ValueError("No samples provided to build batch result")

    if len(samples) != num_generations:
        raise ValueError(
            f"Expected {num_generations} samples in the group, received {len(samples)}"
        )

    prompts_list: List[Any] = []
    completions_list: List[Any] = []
    rewards_list: List[float] = []
    rewards_missing = False

    for entry in samples:
        prompt_value = entry.get("input", entry.get("prompt", ""))
        completion_value = entry.get("completion", entry.get("answer", ""))

        prompts_list.append(copy.deepcopy(prompt_value))
        completions_list.append(copy.deepcopy(completion_value))

        reward_value = entry.get("reward")
        if isinstance(reward_value, (list, tuple)):
            if len(reward_value) != 1:
                raise ValueError(
                    "Reward lists must contain exactly one value when treating a single group"
                )
            reward_value = reward_value[0]

        if reward_value is None:
            rewards_missing = True
        else:
            rewards_list.append(float(reward_value))

    batch_rewards = None if rewards_missing else rewards_list

    batch_result = build_batch_result(
        tokenizer=tokenizer,
        batch_id=batch_id,
        prompts=prompts_list,
        completions=completions_list,
        rewards=batch_rewards,
        max_prompt_length=max_prompt_length,
        max_seq_len=max_seq_len,
    )

    return batch_result

def build_batch_result(
    *,
    tokenizer: Any,
    batch_id: int,
    prompts: Iterable[Any],
    completions: Iterable[Any],
    rewards: Iterable[float] | None,
    max_prompt_length: int | None,
    max_seq_len: int,
) -> BatchResult:
    """Create a ``BatchResult`` from pre-tokenized prompt/completion pairs."""

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    effective_max_prompt = max_prompt_length or max_seq_len

    prompts_list = list(prompts)
    completions_list = list(completions)
    if len(prompts_list) != len(completions_list):
        raise ValueError(
            f"Prompt/completion count mismatch: {len(prompts_list)} vs {len(completions_list)}"
        )

    reward_list = list(rewards) if rewards is not None else None
    if reward_list is not None and len(reward_list) != len(prompts_list):
        raise ValueError(
            f"Reward count mismatch: expected {len(prompts_list)}, got {len(reward_list)}"
        )

    prompt_ids: List[List[int]] = []
    prompt_mask: List[List[int]] = []
    completion_ids: List[List[int]] = []
    completion_mask: List[List[int]] = []
    completion_logprobs: List[List[float]] = []
    reward_values: List[float] = []

    for idx, (prompt_obj, completion_obj) in enumerate(zip(prompts_list, completions_list)):
        prompt_text = _normalize_prompt(prompt_obj)
        completion_text = _normalize_completion(completion_obj)

        prompt_tokens = _tokenize(tokenizer, prompt_text)
        if effective_max_prompt is not None and len(prompt_tokens) > effective_max_prompt:
            prompt_tokens = prompt_tokens[-effective_max_prompt:]
        if not prompt_tokens:
            if pad_token_id is not None:
                prompt_tokens = [pad_token_id]
            elif eos_token_id is not None:
                prompt_tokens = [eos_token_id]
            else:
                prompt_tokens = [0]

        available_for_completion = max(1, max_seq_len - len(prompt_tokens))
        completion_tokens = _tokenize(tokenizer, completion_text)[:available_for_completion]
        if eos_token_id is not None:
            completion_tokens = completion_tokens + [eos_token_id]
        if not completion_tokens:
            if pad_token_id is not None:
                completion_tokens = [pad_token_id]
            elif eos_token_id is not None:
                completion_tokens = [eos_token_id]
            else:
                completion_tokens = [0]

        prompt_ids.append(prompt_tokens)
        prompt_mask.append([1] * len(prompt_tokens))
        completion_ids.append(completion_tokens)
        completion_mask.append([1] * len(completion_tokens))
        completion_logprobs.append([0.0] * len(completion_tokens))

        if reward_list is not None:
            reward_value = float(reward_list[idx])
        else:
            reward_value = float(len(completion_tokens))
        reward_values.append(reward_value)

    processed = ProcessedOutputs(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        completion_logprobs=completion_logprobs,
        rewards=reward_values,
    )

    return BatchResult(
        batch_id=batch_id,
        processed_results=processed,
        generation_time=0.0,
        all_reward_dict={"reward": reward_values},
        completions=[_normalize_completion(c) for c in completions_list],
        prompts=[_normalize_prompt(p) for p in prompts_list],
    )