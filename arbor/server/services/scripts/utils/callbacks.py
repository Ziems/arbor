import logging
import time

from transformers import TrainerCallback

from arbor.server.services.comms.comms import ArborScriptCommsHandler
from arbor.server.services.scripts.utils.ingestion_monitor import IngestionMonitor

logger = logging.getLogger(__name__)


class WeightUpdateCallback(TrainerCallback):
    """A callback that coordinates weight updates with the server using a handshake mechanism"""

    def __init__(self, ingestion_monitor: IngestionMonitor):
        self.comms_handler = None
        self.trainer = None
        self.ingestion_monitor = ingestion_monitor
        self._waiting_for_ready = False

    def set_comms_handler(self, comms_handler: ArborScriptCommsHandler):
        self.comms_handler = comms_handler

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        self.ingestion_monitor.set_last_step_time()

        # Check if we need to do a weight update (on all processes)
        if self.trainer and state.global_step != self.trainer._last_loaded_step:
            # Only the main process handles server coordination
            if self.comms_handler and self.comms_handler.is_main_process:
                logger.info("Starting weight update coordination with server...")

                # Step 1: Request permission to update weights
                self.comms_handler.send_status({"status": "weight_update_request"})
                self._waiting_for_ready = True

                # Step 2: Wait for server to signal it's ready (all inference requests finished)
                logger.info(
                    "Waiting for server to confirm inference requests are complete..."
                )
                max_wait_time = 60  # Maximum time to wait for server response
                wait_start = time.time()

                while self._waiting_for_ready:
                    if time.time() - wait_start > max_wait_time:
                        logger.warning(
                            f"Timeout waiting for server ready signal after {max_wait_time}s, proceeding anyway..."
                        )
                        break
                    time.sleep(0.1)  # Check frequently for server response

                logger.info(
                    "Server ready, all processes will now perform weight update..."
                )

            # All processes need to synchronize before weight update
            if hasattr(self.trainer, "accelerator"):
                self.trainer.accelerator.wait_for_everyone()

            # Step 3: Perform the actual weight update (ON ALL PROCESSES)
            logger.info(
                f"Process rank {getattr(self.trainer.accelerator, 'process_index', 'unknown')} - About to call _move_model_to_vllm() for step {state.global_step}"
            )

            try:
                import signal

                def timeout_handler(signum, frame):
                    raise TimeoutError(
                        "_move_model_to_vllm() call timed out after 120 seconds"
                    )

                # Set a 2-minute timeout for the weight update
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(120)

                logger.info(
                    f"Process rank {getattr(self.trainer.accelerator, 'process_index', 'unknown')} - Starting _move_model_to_vllm() call..."
                )
                start_time = time.time()
                self.trainer._move_model_to_vllm()
                end_time = time.time()
                logger.info(
                    f"Process rank {getattr(self.trainer.accelerator, 'process_index', 'unknown')} - Successfully completed _move_model_to_vllm() call in {end_time - start_time:.2f} seconds"
                )
                self.trainer._last_loaded_step = state.global_step

                # Cancel the timeout
                signal.alarm(0)

            except Exception as e:
                # Cancel the timeout on error
                signal.alarm(0)
                logger.error(
                    f"Process rank {getattr(self.trainer.accelerator, 'process_index', 'unknown')} - Error during _move_model_to_vllm(): {e}"
                )
                logger.error(f"Exception type: {type(e).__name__}")
                import traceback

                logger.error(f"Traceback: {traceback.format_exc()}")
                raise

            # All processes synchronize after weight update
            if hasattr(self.trainer, "accelerator"):
                self.trainer.accelerator.wait_for_everyone()

            # Only the main process signals completion to server
            if self.comms_handler and self.comms_handler.is_main_process:
                # Step 4: Signal completion to allow inference to resume
                self.comms_handler.send_status({"status": "weight_update_complete"})
                logger.info("Weight update complete, inference can resume")

    def on_command_received(self, command):
        """Handle commands from the server"""
        if command.get("command") == "weight_update_ready":
            logger.info("Received ready signal from server")
            self._waiting_for_ready = False

    def on_save(
        self,
        args,  #: TrainingArguments,
        state,  #: TrainerState,
        control,  #: TrainerControl,
        **kwargs,
    ):
        self.comms_handler.send_status(
            {
                "status": "checkpoint_saved",
                "checkpoint_name": self.trainer.last_checkpoint_name,
                "output_dir": self.trainer.checkpoint_dir
                + f"/{self.trainer.last_checkpoint_name}/",
            }
        )
