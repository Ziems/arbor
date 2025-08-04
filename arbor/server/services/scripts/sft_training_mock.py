# Mock version of sft_training.py for testing purposes
import argparse
import json
import signal
import sys
import time


# Mock classes to simulate SFT training without GPU dependencies
class MockSFTTrainer:
    def __init__(self, model, args=None, **kwargs):
        self.model = model
        self.args = args
        print(f"Mock: Initialized SFTTrainer for model {model}")

    def train(self):
        print("Mock: Starting SFT training...")
        for i in range(4):  # Mock 4 training steps
            print(f"Mock: SFT training step {i+1}/4")
            time.sleep(0.1)  # Simulate training time
        print("Mock: SFT training completed!")


class MockSFTConfig:
    def __init__(self, **kwargs):
        self.dataloader_num_workers = kwargs.get("dataloader_num_workers", 0)
        self.output_dir = kwargs.get("output_dir", ".")
        for key, value in kwargs.items():
            setattr(self, key, value)
        print(f"Mock: Created SFTConfig with output_dir={self.output_dir}")


class MockLoraConfig:
    def __init__(self, **kwargs):
        print("Mock: Created LoraConfig for SFT")


class MockCommsHandler:
    def __init__(self, **kwargs):
        print("Mock: Created ArborScriptCommsHandler for SFT")

    def close(self):
        print("Mock: Closing SFT comms handler")


def get_mock_training_arg_parser():
    """Mock version of get_training_arg_parser"""
    parser = argparse.ArgumentParser()

    pipe_args = parser.add_argument_group("Comms arguments")
    pipe_args.add_argument("--host", default="localhost")
    pipe_args.add_argument("--command_port", type=int)
    pipe_args.add_argument("--status_port", type=int)
    pipe_args.add_argument("--data_port", type=int)
    pipe_args.add_argument("--broadcast_port", type=int)
    pipe_args.add_argument("--handshake_port", type=int)

    training_args = parser.add_argument_group("Training arguments")
    training_args.add_argument("--model", type=str, help="Model to use for training")
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

    return parser


def main():
    parser = get_mock_training_arg_parser()
    args, unknown = parser.parse_known_args()

    print("Mock SFT Training Script Starting...")
    print(f"Model: {args.model}")
    print("This is a mock SFT training script - no actual GPU operations will occur")

    if unknown:
        print(f"Mock: Ignoring unknown args: {unknown}")

    try:
        trl_train_args = {**(args.trl_train_kwargs or {})}
        arbor_train_args = {**(args.arbor_train_kwargs or {})}

        print(f"Mock: TRL args: {trl_train_args}")
        print(f"Mock: Arbor args: {arbor_train_args}")

        # Ensure output_dir is present
        if "output_dir" not in trl_train_args:
            trl_train_args["output_dir"] = "."
            print("Mock: Setting default output_dir")

        # Mock LORA config if needed
        lora_config = None
        if arbor_train_args.get("lora", False):
            print("Mock: Using LORA for PEFT in SFT")
            lora_config = MockLoraConfig()

        # Handle gradient checkpointing for LORA
        if "gradient_checkpointing_kwargs" in trl_train_args and arbor_train_args.get(
            "lora", False
        ):
            print(
                "Mock: Setting gradient_checkpointing_kwargs to use_reentrant=False for LORA training"
            )
            trl_train_args["gradient_checkpointing_kwargs"] = {
                **(trl_train_args.get("gradient_checkpointing_kwargs") or {}),
                "use_reentrant": False,
            }

        training_args = MockSFTConfig(
            dataloader_num_workers=0,
            **trl_train_args,
        )

        trainer = MockSFTTrainer(
            model=args.model,
            args=training_args,
            peft_config=lora_config,
        )

        # Create mock comms handler
        comms_handler = MockCommsHandler(
            host=args.host,
            command_port=args.command_port,
            status_port=args.status_port,
            data_port=args.data_port,
            broadcast_port=args.broadcast_port,
            handshake_port=args.handshake_port,
        )

        # Mock signal handlers
        def signal_handler(signum, frame):
            print(f"\nMock: SFT received signal {signum}. Shutting down...")
            comms_handler.close()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        print("Mock: Starting SFT training...")
        try:
            trainer.train()
        except Exception as e:
            print(f"Mock: Error during SFT training: {e}")
            raise

    except KeyboardInterrupt:
        print("\nMock: SFT received interrupt, shutting down...")
    except Exception as e:
        print(f"Mock: SFT Error: {e}")
        raise e
    finally:
        print("Mock: SFT cleaning up resources...")
        if "comms_handler" in locals():
            comms_handler.close()
        print("Mock: SFT cleanup complete")


if __name__ == "__main__":
    main()
