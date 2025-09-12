import os
import shutil
import threading
import time
from typing import Callable, Optional

from peft import AutoPeftModelForCausalLM
from transformers import Trainer

from arbor.server.services.comms.comms import ArborScriptCommsHandler


class CommandMonitor:
    def __init__(
        self,
        comms_handler: ArborScriptCommsHandler,
        trainer: Trainer,
        base_model_name: str,
        ingestion_monitor: Optional["IngestionMonitor"] = None,
        weight_update_callback=None,
    ):
        self.comms_handler = comms_handler
        self.trainer = trainer
        self.base_model_name = base_model_name
        self.command_thread = threading.Thread(
            target=self._monitor_commands, daemon=True
        )
        self.ingestion_monitor = ingestion_monitor
        self.weight_update_callback = weight_update_callback

    def start(self):
        self.command_thread.start()

    def _monitor_commands(self):
        """Background thread that monitors for commands from the server."""
        if not self.comms_handler:
            return
        try:
            for command in self.comms_handler.receive_command():
                print(f"Main process received command: {command}")
                if command.get("command") == "save_model":
                    self.trainer.args.output_dir = self.trainer.save_model_dir
                    self.trainer.state.should_save = True
                elif command.get("command") == "save_checkpoint":
                    self.trainer.args.output_dir = (
                        self.trainer.checkpoint_dir
                        + f"/{command.get('checkpoint_name')}/"
                    )
                    self.trainer.state.should_save = True
                    self.trainer.last_checkpoint_name = command.get("checkpoint_name")

                elif command.get("command") == "weight_update_ready":
                    # Forward to weight update callback
                    if self.weight_update_callback:
                        self.weight_update_callback.on_command_received(command)
                elif command.get("command") == "terminate":
                    print("TERMINATED")
                    self.trainer.accelerator.end_training()
                    self.comms_handler.send_status({"status": "terminated"})

        except Exception as e:
            print(e)
            self.comms_handler.send_status({"status": "error", "error": str(e)})
