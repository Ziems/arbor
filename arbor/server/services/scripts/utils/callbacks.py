import logging

from transformers import TrainerCallback

from arbor.server.services.comms.comms import ArborScriptCommsHandler
from arbor.server.services.scripts.utils.ingestion_monitor import IngestionMonitor

logger = logging.getLogger(__name__)


class WeightUpdateCallback(TrainerCallback):
    """A callback that sends weight update completion status after each step"""

    def __init__(self, ingestion_monitor: IngestionMonitor):
        self.comms_handler = None
        self.trainer = None
        self.ingestion_monitor = ingestion_monitor

    def set_comms_handler(self, comms_handler: ArborScriptCommsHandler):
        self.comms_handler = comms_handler

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        self.ingestion_monitor.set_last_step_time()
        if self.comms_handler and self.comms_handler.is_main_process and self.trainer:
            if state.global_step != self.trainer._last_loaded_step:
                logger.info("Updating inference model...")
                self.comms_handler.send_status({"status": "weight_update_start"})
                self.trainer._move_model_to_vllm()
                self.trainer._last_loaded_step = state.global_step
                self.comms_handler.send_status({"status": "weight_update_complete"})
