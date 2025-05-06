import argparse
import json

from trl import SFTConfig, SFTTrainer


class ArborSFTTrainer(SFTTrainer):
    pass


def dataset_from_file(data_path):
    """
    Creates a HuggingFace Dataset from a JSONL file.
    """
    from datasets import load_dataset

    dataset = load_dataset("json", data_files=data_path, split="train")
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")

    pipe_args = parser.add_argument_group("Comms arguments")
    pipe_args.add_argument("--host", default="localhost")
    pipe_args.add_argument("--command_port", type=int, required=True)
    pipe_args.add_argument("--status_port", type=int, required=True)
    pipe_args.add_argument("--data_port", type=int, required=True)
    pipe_args.add_argument("--broadcast_port", type=int, required=True)

    training_args = parser.add_argument_group("Training arguments")
    training_args.add_argument(
        "--model",
        type=str,
        help="Model to use for training",
    )
    training_args.add_argument(
        "--train_data_path",
        type=str,
        help="Path to the training data",
    )
    training_args.add_argument(
        "--trl_config_kwargs",
        type=json.loads,
        help="Training arguments as a JSON string",
    )
    training_args.add_argument(
        "--trainer_kwargs",
        type=json.loads,
        help="Training arguments as a JSON string",
    )

    args = parser.parse_args()

    try:
        trl_config_args = {**(args.trl_config_kwargs or {})}
        trainer_args = {**(args.trainer_kwargs or {})}
        trl_config_args["bf16"] = True

        # TODO: These assertions should be done in some better way
        assert "output_dir" in trl_config_args, "output_dir is required"
        assert "train_data_path" in trainer_args, "train_data_path is required"

        config = SFTConfig(**trl_config_args)
        trainer = ArborSFTTrainer(
            model=args.model,
            trl_config=config,
            **trainer_args,
        )
    except Exception as e:
        print(f"Error: {e}")
        raise e
