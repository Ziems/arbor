# Mock version of grpo_training.py for testing purposes
import argparse
import json
import random
import signal
import sys
import threading
import time
from typing import Any, List, Optional, Union


# Mock the GPU-related imports
class MockTorch:
    class cuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def device_count():
            return 0

    @staticmethod
    def tensor(*args, **kwargs):
        return MockTensor()

    @staticmethod
    def no_grad():
        return MockContextManager()

    @staticmethod
    def cat(*args, **kwargs):
        return MockTensor()


class MockTensor:
    def __init__(self, *args, **kwargs):
        self.shape = (1, 1)  # Mock shape

    def to(self, device):
        return self

    def sum(self, *args, **kwargs):
        return MockTensor()

    def mean(self, *args, **kwargs):
        return MockTensor()

    def item(self):
        return 0.5  # Mock scalar value

    def tolist(self):
        return [0.5]


class MockContextManager:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# Mock accelerate and distributed operations
class MockAccelerator:
    def __init__(self):
        self.device = "cpu"
        self.is_main_process = True
        self.process_index = 0

    def wait_for_everyone(self):
        print("Mock: Waiting for everyone")

    def gather(self, tensor):
        return tensor

    def unwrap_model(self, model):
        return MockModel()

    def end_training(self):
        print("Mock: Ending training")


class MockModel:
    def __init__(self):
        self.training = True

    def disable_adapter(self):
        return MockContextManager()


class MockVLLMClient:
    def __init__(self, host, port, group_port=None, connection_timeout=None):
        self.host = host
        self.port = port
        print(f"Mock: Initializing vLLM client to {host}:{port}")

    def init_communicator(self):
        print("Mock: Initializing vLLM communicator")


class MockGRPOTrainer:
    def __init__(self, *args, **kwargs):
        self.accelerator = MockAccelerator()
        self.model = MockModel()
        self.processing_class = MockTokenizer()
        self.state = MockTrainerState()
        self._metrics = {"train": self._init_metrics(), "eval": self._init_metrics()}
        self._textual_logs = {"prompt": [], "completion": [], "advantages": []}
        self.max_prompt_length = None
        self.max_completion_length = None
        self.mask_truncated_completions = False
        self.num_generations = 1
        self.beta = 0.1
        self.ref_model = None
        self.num_iterations = 1
        print("Mock: Initialized GRPO Trainer")

    def _init_metrics(self):
        return {
            "num_tokens": [],
            "completions/mean_length": [],
            "completions/min_length": [],
            "completions/max_length": [],
            "completions/clipped_ratio": [],
            "completions/mean_terminated_length": [],
            "completions/min_terminated_length": [],
            "completions/max_terminated_length": [],
            "reward": [],
            "reward_std": [],
            "frac_reward_zero_std": [],
        }

    def train(self):
        print("Mock: Starting training...")
        for i in range(5):  # Mock 5 training steps
            print(f"Mock: Training step {i+1}/5")
            time.sleep(0.1)  # Simulate training time
        print("Mock: Training completed!")

    def _get_per_token_logps(self, model, ids, mask, logits_to_keep, batch_size=None):
        # Mock per-token log probabilities
        return MockTensor()


class MockTrainerState:
    def __init__(self):
        self.num_input_tokens_seen = 0


class MockTokenizer:
    def __init__(self):
        self.eos_token_id = 2

    def __call__(self, *args, **kwargs):
        return {"input_ids": MockTensor(), "attention_mask": MockTensor()}


class MockGRPOConfig:
    def __init__(self, **kwargs):
        self.use_vllm = True
        self.vllm_server_host = kwargs.get("vllm_server_host", "localhost")
        self.vllm_server_port = kwargs.get("vllm_server_port", 8000)
        self.vllm_server_timeout = kwargs.get("vllm_server_timeout", 30)
        self.vllm_guided_decoding_regex = kwargs.get("vllm_guided_decoding_regex", None)
        for key, value in kwargs.items():
            setattr(self, key, value)


class MockArborGRPOTrainer(MockGRPOTrainer):
    def __init__(self, model, args=None, vllm_group_port=None, **kwargs):
        super().__init__()
        self.vllm_client = None
        if self.accelerator.is_main_process:
            print(
                f"Mock: Creating vLLM client for port {args.vllm_server_port if args else 8000}"
            )
            self.vllm_client = MockVLLMClient(
                args.vllm_server_host if args else "localhost",
                args.vllm_server_port if args else 8000,
                group_port=vllm_group_port,
            )
            self.vllm_client.init_communicator()


class MockDataset:
    def __init__(self, ingestion_monitor=None):
        self.ingestion_monitor = ingestion_monitor
        self.comms_handler = None
        self.accelerator = None
        print("Mock: Created BlockingRotatingQueueDataset")

    def set_comms_handler(self, handler):
        self.comms_handler = handler

    def set_accelerator(self, accelerator):
        self.accelerator = accelerator


class MockWeightUpdateCallback:
    def __init__(self, ingestion_monitor=None):
        self.ingestion_monitor = ingestion_monitor
        self.comms_handler = None
        self.trainer = None
        print("Mock: Created WeightUpdateCallback")

    def set_comms_handler(self, handler):
        self.comms_handler = handler

    def set_trainer(self, trainer):
        self.trainer = trainer


class MockLastStepTimeCallback:
    def __init__(self, ingestion_monitor):
        self.ingestion_monitor = ingestion_monitor
        print("Mock: Created LastStepTimeCallback")


class MockIngestionMonitor:
    def time_since_last_step(self):
        return 1.0  # Mock time

    def set_last_step_time(self):
        pass


class MockCommsHandler:
    def __init__(self, **kwargs):
        print("Mock: Created ArborScriptCommsHandler")

    def send_status(self, status):
        print(f"Mock: Sending status: {status}")

    def close(self):
        print("Mock: Closing comms handler")


class MockCommandMonitor:
    def __init__(self, **kwargs):
        print("Mock: Created CommandMonitor")

    def start(self):
        print("Mock: Starting command monitor")


class MockLoraConfig:
    def __init__(self, **kwargs):
        print("Mock: Created LoraConfig")


def mock_load_dataset(name, split=None):
    """Mock dataset loader"""
    print(f"Mock: Loading dataset {name} split {split}")
    return MockDatasetItems()


class MockDatasetItems:
    def __iter__(self):
        # Generate mock data items
        for i in range(10):  # Mock 10 items
            yield {"prompt": f"This is mock prompt {i}", "content": f"Mock content {i}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")

    pipe_args = parser.add_argument_group("Comms arguments")
    pipe_args.add_argument("--host", default="localhost")
    pipe_args.add_argument("--command_port", type=int, required=True)
    pipe_args.add_argument("--status_port", type=int, required=True)
    pipe_args.add_argument("--data_port", type=int, required=True)
    pipe_args.add_argument("--broadcast_port", type=int, required=True)
    pipe_args.add_argument("--handshake_port", type=int, required=True)
    pipe_args.add_argument("--vllm_group_port", type=int, required=True)
    pipe_args.add_argument("--vllm_port", type=int, required=True)

    training_args = parser.add_argument_group("Training arguments")
    training_args.add_argument(
        "--model",
        type=str,
        help="Model to use for training",
    )
    training_args.add_argument(
        "--trl_train_kwargs",
        type=json.loads,
        help="Training arguments as a JSON string",
    )
    training_args.add_argument(
        "--arbor_train_kwargs",
        type=json.loads,
        help="Training arguments as a JSON string",
    )

    args = parser.parse_args()

    print("Mock GRPO Training Script Starting...")
    print(f"Model: {args.model}")
    print("This is a mock training script - no actual GPU operations will occur")

    try:
        trl_train_args = {**(args.trl_train_kwargs or {})}
        arbor_train_args = {**(args.arbor_train_kwargs or {})}

        print(f"Mock: TRL args: {trl_train_args}")
        print(f"Mock: Arbor args: {arbor_train_args}")

        # Mock LORA config if needed
        lora_config = None
        if arbor_train_args.get("lora", False):
            print("Mock: Using LORA for PEFT")
            lora_config = MockLoraConfig()

        # Mock wandb initialization
        if "report_to" in trl_train_args and trl_train_args["report_to"] == "wandb":
            print("Mock: Would initialize wandb here")

        training_args = MockGRPOConfig(
            dataloader_num_workers=0,
            shuffle_dataset=False,
            vllm_server_port=args.vllm_port,
            **trl_train_args,
        )

        # Create mock components
        ingestion_monitor = MockIngestionMonitor()
        train_dataset = MockDataset(ingestion_monitor=ingestion_monitor)
        weight_update_callback = MockWeightUpdateCallback(
            ingestion_monitor=ingestion_monitor
        )

        trainer = MockArborGRPOTrainer(
            model=args.model,
            args=training_args,
            train_dataset=train_dataset,
            callbacks=[
                MockLastStepTimeCallback(ingestion_monitor),
                weight_update_callback,
            ],
            peft_config=lora_config,
            vllm_group_port=args.vllm_group_port,
        )

        # Create mock comms handler
        comms_handler = MockCommsHandler(
            host=args.host,
            command_port=args.command_port,
            status_port=args.status_port,
            data_port=args.data_port,
            broadcast_port=args.broadcast_port,
            handshake_port=args.handshake_port,
            is_main_process=trainer.accelerator.is_main_process,
        )

        train_dataset.set_comms_handler(comms_handler)
        train_dataset.set_accelerator(trainer.accelerator)

        weight_update_callback.set_comms_handler(comms_handler)
        weight_update_callback.set_trainer(trainer)
        trainer.comms_handler = comms_handler

        command_monitor = MockCommandMonitor(
            comms_handler=comms_handler,
            trainer=trainer,
            base_model_name=args.model,
            ingestion_monitor=ingestion_monitor,
        )
        command_monitor.start()

        # Mock signal handlers
        def signal_handler(signum, frame):
            print(f"\nMock: Received signal {signum}. Shutting down...")
            comms_handler.close()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        print("Mock: Starting training...")
        try:
            trainer.train()
        except Exception as e:
            print(f"Mock: Error during training: {e}")
            raise

    except KeyboardInterrupt:
        print("\nMock: Received interrupt, shutting down...")
    except Exception as e:
        print(f"Mock: Error: {e}")
        if "comms_handler" in locals():
            comms_handler.send_status({"status": "error", "error": str(e)})
        raise e
    finally:
        print("Mock: Cleaning up resources...")
        if "trainer" in locals():
            trainer.accelerator.end_training()
        if "comms_handler" in locals():
            comms_handler.close()
        print("Mock: Cleanup complete")


if __name__ == "__main__":
    main()
