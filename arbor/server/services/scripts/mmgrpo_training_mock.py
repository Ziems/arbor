# Mock version of mmgrpo_training.py for testing purposes
import argparse
import json
import signal
import sys
import time
from typing import Any, Optional, Union

# Reuse mock classes from grpo_training_mock for consistency
from .grpo_training_mock import (
    MockAccelerator,
    MockCommandMonitor,
    MockCommsHandler,
    MockDataset,
    MockGRPOConfig,
    MockGRPOTrainer,
    MockIngestionMonitor,
    MockLoraConfig,
    MockModel,
    MockTensor,
    MockTokenizer,
    MockTorch,
    MockTrainerState,
    MockVLLMClient,
    MockWeightUpdateCallback,
    mock_load_dataset,
)


class MockMMGRPOTrainer(MockGRPOTrainer):
    """Mock Multi-Modal GRPO Trainer"""

    def __init__(
        self,
        model,
        args=None,
        lora=False,
        vllm_group_port=None,
        max_context_length=None,
        grpo_flavor="mmgrpo",
        **kwargs,
    ):
        super().__init__()
        self.lora = lora
        self.max_context_length = max_context_length
        self.grpo_flavor = grpo_flavor
        self.vllm_client = None

        print(f"Mock: Initializing MMGRPOTrainer with flavor={grpo_flavor}")

        if self.accelerator.is_main_process:
            print(
                f"Mock: Creating vLLM client for MMGRPO port {args.vllm_server_port if args else 8000}"
            )
            self.vllm_client = MockVLLMClient(
                args.vllm_server_host if args else "localhost",
                args.vllm_server_port if args else 8000,
                group_port=vllm_group_port,
            )
            self.vllm_client.init_communicator()

    def train(self):
        print("Mock: Starting MM-GRPO training...")
        for i in range(3):  # Mock 3 training steps for MM-GRPO
            print(f"Mock: MM-GRPO training step {i+1}/3")
            time.sleep(0.1)  # Simulate training time
        print("Mock: MM-GRPO training completed!")


class MockBlockingQueueDataset(MockDataset):
    """Mock version of BlockingQueueDataset specifically for MMGRPO"""

    def __init__(self, ingestion_monitor=None):
        super().__init__(ingestion_monitor)
        print("Mock: Created BlockingQueueDataset for MMGRPO")


def main():
    parser = argparse.ArgumentParser()

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

    print("Mock MM-GRPO Training Script Starting...")
    print(f"Model: {args.model}")
    print(
        "This is a mock multi-modal training script - no actual GPU operations will occur"
    )

    try:
        trl_train_args = {**(args.trl_train_kwargs or {})}
        arbor_train_args = {**(args.arbor_train_kwargs or {})}

        print(f"Mock: TRL args: {trl_train_args}")
        print(f"Mock: Arbor args: {arbor_train_args}")

        # Mock LORA config if needed
        lora_config = None
        lora = arbor_train_args.get("lora", False)
        if lora:
            print("Mock: Using LORA for PEFT in MMGRPO")
            lora_config = MockLoraConfig()

        training_args = MockGRPOConfig(
            dataloader_num_workers=0,
            shuffle_dataset=False,
            vllm_server_port=args.vllm_port,
            **trl_train_args,
        )

        # Create mock components specific to MMGRPO
        ingestion_monitor = MockIngestionMonitor()
        train_dataset = MockBlockingQueueDataset(ingestion_monitor=ingestion_monitor)
        weight_update_callback = MockWeightUpdateCallback(
            ingestion_monitor=ingestion_monitor
        )

        trainer = MockMMGRPOTrainer(
            model=args.model,
            args=training_args,
            train_dataset=train_dataset,
            callbacks=[weight_update_callback],
            peft_config=lora_config,
            lora=lora,
            vllm_group_port=args.vllm_group_port,
            max_context_length=arbor_train_args.get("max_context_length"),
            grpo_flavor="mmgrpo",
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

        command_monitor = MockCommandMonitor(
            comms_handler=comms_handler,
            trainer=trainer,
            base_model_name=args.model,
            ingestion_monitor=ingestion_monitor,
        )
        command_monitor.start()

        # Mock signal handlers
        def signal_handler(signum, frame):
            print(f"\nMock: Received signal {signum}. Shutting down MM-GRPO...")
            comms_handler.close()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        print("Mock: Starting MM-GRPO training...")
        try:
            trainer.train()
        except Exception as e:
            print(f"Mock: Error during MM-GRPO training: {e}")
            raise

    except KeyboardInterrupt:
        print("\nMock: MM-GRPO received interrupt, shutting down...")
    except Exception as e:
        print(f"Mock: MM-GRPO Error: {e}")
        if "comms_handler" in locals():
            comms_handler.send_status({"status": "error", "error": str(e)})
        raise e
    finally:
        print("Mock: MM-GRPO cleaning up resources...")
        if "trainer" in locals():
            trainer.accelerator.end_training()
        if "comms_handler" in locals():
            comms_handler.close()
        print("Mock: MM-GRPO cleanup complete")


if __name__ == "__main__":
    main()
