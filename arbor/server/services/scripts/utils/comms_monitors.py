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
                    print(
                        f"[Training Script] Process {getattr(self.trainer.accelerator, 'process_index', 0)} instructed to save model at {self.trainer.args.output_dir}"
                    )

                    # Wait for ingestion to finish (all processes should wait)
                    while self.ingestion_monitor and (
                        self.ingestion_monitor.time_since_last_step() <= 10
                        or self.ingestion_monitor.time_since_last_queue_pop() <= 10
                    ):
                        print(
                            f"Process {getattr(self.trainer.accelerator, 'process_index', 0)}: Waiting for steps to finish"
                        )
                        if self.ingestion_monitor:
                            print(
                                f"Time since last step: {self.ingestion_monitor.time_since_last_step():.1f} (needs to be >= 10)"
                            )
                            print(
                                f"Time since last queue pop: {self.ingestion_monitor.time_since_last_queue_pop():.1f} (needs to be >= 10)"
                            )
                        time.sleep(5)

                    print(
                        f"[Training Script] Process {getattr(self.trainer.accelerator, 'process_index', 0)} saving model..."
                    )

                    # For DeepSpeed and distributed training, all processes need to participate in save_model
                    # The trainer.save_model() method handles the coordination internally
                    if self.trainer.peft_config:
                        self.trainer.save_model(
                            output_dir=self.trainer.args.output_dir + "/adapter/"
                        )

                        # Wait for all processes to finish saving before merge (DeepSpeed requirement)
                        if hasattr(self.trainer.accelerator, "wait_for_everyone"):
                            self.trainer.accelerator.wait_for_everyone()

                        # Only main process does the merge and unload to avoid conflicts
                        if getattr(self.trainer.accelerator, "is_main_process", True):
                            print(
                                "[Training Script] Main process performing PEFT merge..."
                            )
                            _model_to_merge = AutoPeftModelForCausalLM.from_pretrained(
                                self.trainer.args.output_dir + "/adapter/",
                                config=self.trainer.peft_config,
                            )
                            merged_model = _model_to_merge.merge_and_unload()
                            merged_model.save_pretrained(
                                self.trainer.args.output_dir,
                                safe_serialization=True,
                            )
                            self.trainer.processing_class.save_pretrained(
                                self.trainer.args.output_dir
                            )
                    else:
                        # For non-PEFT models, let all processes participate in saving
                        # This is critical for DeepSpeed which needs all processes to save their shards
                        self.trainer.save_model()

                        # Wait for all processes to complete saving
                        if hasattr(self.trainer.accelerator, "wait_for_everyone"):
                            self.trainer.accelerator.wait_for_everyone()

                    # Only main process sends status to avoid duplicate messages
                    if getattr(self.trainer.accelerator, "is_main_process", True):
                        print("[Training Script] Model saved successfully")
                        self.comms_handler.send_status(
                            {
                                "status": "model_saved",
                                "output_dir": self.trainer.args.output_dir,
                            }
                        )
                elif command.get("command") == "save_checkpoint":
                    print(
                        f"[Training Script] Instructed to save checkpoint {command.get('checkpoint_name')}"
                    )
                    while self.ingestion_monitor and (
                        self.ingestion_monitor.time_since_last_step() <= 10
                        or self.ingestion_monitor.time_since_last_queue_pop() <= 10
                    ):
                        print(f"Waiting for steps to finish")
                        if self.ingestion_monitor:
                            print(
                                f"Time since last step: {self.ingestion_monitor.time_since_last_step():.1f} (needs to be >= 10)"
                            )
                            print(
                                f"Time since last queue pop: {self.ingestion_monitor.time_since_last_queue_pop():.1f} (needs to be >= 10)"
                            )
                        time.sleep(5)
                    if self.trainer.peft_config:
                        self.trainer.save_model(
                            output_dir=self.trainer.args.output_dir
                            + f"/checkpoints/{command.get('checkpoint_name')}/adapter/"
                        )
                        _model_to_merge = AutoPeftModelForCausalLM.from_pretrained(
                            self.trainer.args.output_dir
                            + f"/checkpoints/{command.get('checkpoint_name')}/adapter/",
                            config=self.trainer.peft_config,
                        )
                        merged_model = _model_to_merge.merge_and_unload()
                        merged_model.save_pretrained(
                            self.trainer.args.output_dir
                            + f"/checkpoints/{command.get('checkpoint_name')}/",
                            safe_serialization=True,
                        )
                        self.trainer.processing_class.save_pretrained(
                            self.trainer.args.output_dir
                            + f"/checkpoints/{command.get('checkpoint_name')}/"
                        )
                    else:
                        self.trainer.save_model(
                            output_dir=self.trainer.args.output_dir
                            + f"/checkpoints/{command.get('checkpoint_name')}/"
                        )

                    # Copy checkpoint files to root output directory
                    checkpoint_dir = (
                        self.trainer.args.output_dir
                        + f"/checkpoints/{command.get('checkpoint_name')}/"
                    )
                    root_dir = self.trainer.args.output_dir

                    # Copy all files from checkpoint dir to root dir, overwriting if they exist
                    # (effectively saves the checkpoint to the output directory)
                    for item in os.listdir(checkpoint_dir):
                        src = os.path.join(checkpoint_dir, item)
                        dst = os.path.join(root_dir, item)
                        if os.path.isdir(src):
                            if os.path.exists(dst):
                                shutil.rmtree(dst)
                            shutil.copytree(src, dst)
                        else:
                            shutil.copy2(src, dst)

                    self.comms_handler.send_status(
                        {
                            "status": "checkpoint_saved",
                            "checkpoint_name": command.get("checkpoint_name"),
                            "output_dir": self.trainer.args.output_dir
                            + f"/checkpoints/{command.get('checkpoint_name')}/",
                        }
                    )
                    self.comms_handler.send_status(
                        {
                            "status": "model_saved",
                            "output_dir": self.trainer.args.output_dir,
                        }
                    )
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
