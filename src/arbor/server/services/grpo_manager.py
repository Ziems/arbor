import random
import string
from datetime import datetime
from pathlib import Path
import os
import subprocess
from multiprocessing import Queue
import multiprocessing

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, setup_chat_format
from trl.data_utils import maybe_apply_chat_template
from trl.models.modeling_base import create_reference_model

from arbor.server.api.models.schemas import GRPORequest, GRPOConfigRequest
from arbor.server.core.config import Settings
from arbor.server.services.job_manager import Job, JobEvent, JobStatus
from arbor.server.services.inference_manager import InferenceManager


class GRPOManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.command_queue = None
        self.status_queue = None
        self.data_queue = None
        self.training_process = None


    def make_output_dir(self, model_name, run_suffix):
        model_name = model_name.split('/')[-1].lower()
        suffix = run_suffix if run_suffix is not None else ''.join(random.choices(string.ascii_letters + string.digits, k=6))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"grpo:{model_name}:{suffix}:{timestamp}"
        return name, str(Path(self.settings.STORAGE_PATH).resolve() / "models" / name)

    def find_training_args(self, request: GRPOConfigRequest):

        name, output_dir = self.make_output_dir(request.model, request.suffix)

        default_train_kwargs = {
            "device": None,
            "use_peft": False,
            "num_train_epochs": 5,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 1e-5,
            "max_seq_length": None,
            "packing": False, # TODO: Turning this off for now
            "bf16": True,
            "output_dir": output_dir,
            # Using the default TRL values here
            "beta": 0.04,
            "num_iterations": 1,
            "temperature": 0.9, # TODO: This should match vLLM
            "num_generations": 8,
            "scale_rewards": True,
            "epsilon_low": 0.2,
            "epsilon_high": 0.2,
            "update_interval": 25
        }

        train_kwargs = request.model_dump(exclude_unset=True)
        train_kwargs={**default_train_kwargs, **(train_kwargs or {})}

        return train_kwargs

    def initialize(self, request: GRPOConfigRequest, inference_manager: InferenceManager):
        # Create pipes instead of queues
        command_recv, command_send = multiprocessing.Pipe()
        status_recv, status_send = multiprocessing.Pipe()
        data_recv, data_send = multiprocessing.Pipe()

        script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
        script_path = os.path.join(script_dir, "grpo_training.py")

        my_env = os.environ.copy()
        my_env["CUDA_VISIBLE_DEVICES"] = "1"

        # Get file descriptors before passing them
        command_fd = command_recv.fileno()
        status_fd = status_send.fileno()
        data_fd = data_recv.fileno()

        # Pass the appropriate ends to the subprocess
        process = subprocess.Popen(
            [
                "python", "-m", "accelerate.commands.launch",
                script_path,
                "--command_recv", str(command_fd),
                "--status_send", str(status_fd),
                "--data_recv", str(data_fd)
            ],
            env=my_env,
            pass_fds=(command_fd, status_fd, data_fd),
            close_fds=False  # Important: don't close inherited FDs
        )

        # Close the ends we passed to the subprocess
        command_recv.close()
        status_send.close()
        data_recv.close()

        # Keep our ends of the pipes
        self.command_send = command_send  # To send commands TO the training process
        self.status_recv = status_recv    # To receive status FROM the training process
        self.data_send = data_send        # To send data TO the training process

        self.training_process = process

    def initialize_config(self, request: GRPOConfigRequest, inference_manager: InferenceManager):
        self.train_kwargs = self.find_training_args(request)
        self.current_model = request.model

        print("Launching inference server...")
        inference_manager.launch(self.current_model)


    def grpo_step(self, request: GRPORequest, inference_manager: InferenceManager):
        batch = request.batch

        # First check the status queue to see if theres anything we need to handle
        while not self.status_queue.empty():
            status = self.status_queue.get()
            if status["status"] == "update_inference_model":
                inference_manager.update_model(self.train_kwargs["output_dir"])
                self.current_model = self.train_kwargs["output_dir"]
            else:
                print("Unhandled status:")
                print(status)

        # This data queue is read by the training process
        self.data_queue.put(batch)

        return self.current_model

    def terminate(self, inference_manager: InferenceManager):
        if inference_manager.process is not None:
            inference_manager.kill()

        output_dir = self.train_kwargs.get("output_dir")
        if output_dir is None:
            print("No output directory specified in train_kwargs, skipping model save")
            return

        print(f"Saving model to {output_dir}")
        return self.model.save_pretrained(output_dir, from_pt=True)

