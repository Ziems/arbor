# Assumes that the server is running
import requests
from datasets import load_dataset
from openai import OpenAI

arbor_port = 8234

client = OpenAI(
    base_url=f"http://127.0.0.1:{arbor_port}/v1",  # Using Arbor server
    api_key="not-needed",  # If you're using a local server, you dont need an API key
)

num_generations = 4
grad_accum_steps = 40
context_length = 8000
current_model =  "qwen/qwen3-8b" # "meta-llama/Llama-3.1-8B-Instruct" #

def initialize_grpo(
    model, url=f"http://127.0.0.1:{arbor_port}/v1/fine_tuning/grpo/initialize"
):
    
    # OptimizerConfig(
    #     optimizer=GRPO,
    #     init_args=dict(
    #         multitask=True,
    #         num_dspy_examples_per_grpo_step=4,
    #         num_rollouts_per_grpo_step=8,
    #         exclude_demos=True,
    #         num_train_steps=750,
    #         num_threads=os.cpu_count(),
    #         use_train_as_val=False,
    #         num_steps_for_val=20,
    #         train_kwargs={
    #             "update_interval": 10,
    #             "per_device_train_batch_size": 1,
    #             "gradient_accumulation_steps": 40,
    #             "temperature": 0.9,
    #             "beta": 0.04,
    #             "learning_rate": 1e-5,
    #             "gradient_checkpointing": True,
    #             "gradient_checkpointing_kwargs": {"use_reentrant": False},
    #             "bf16": True,
    #             "lr_scheduler_type": "constant_with_warmup",
    #             "max_prompt_length": None,
    #             "max_completion_length": None,
    #             "scale_rewards": True,
    #             "max_grad_norm": 0.5,
    #             "lora": True,
    #             # 'report_to': "wandb",
    #             # 'log_completions': True,
    #             # 'logging_steps': 100,
    #             'max_context_length': 6000,
    #         },
    #         report_train_scores=False,
    #         variably_invoked_predictor_grouping_mode="fill",
    #         variably_invoked_predictor_fill_strategy="randint",
    #         grpo_group_size=4,
    #     ),
    #     compile_args=dict(),
    #     langProBe_configs=dict(
    #         use_valset=True,
    #         add_valset_to_trainset=False,
    #         use_model_name_from_optimized_program=True,
    #         set_lm_before_optimizer=True,
    #         launch_arbor=True,
    #     ),
    #     name="GRPO",
    # )

    headers = {"Content-Type": "application/json"}
    data = {"model": model, "num_generations": num_generations, "update_interval": 10, 'report_to': None}

    data.update({
        "update_interval": 10,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": grad_accum_steps,
        "temperature": 0.9,
        "beta": 0.04,
        "learning_rate": 1e-5,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "bf16": True,
        "lr_scheduler_type": "constant_with_warmup",
        "max_prompt_length": None,
        "max_completion_length": None,
        "scale_rewards": True,
        "max_grad_norm": 0.5,
        "lora": True,
        'report_to': None,
        # 'log_completions': True,
        # 'logging_steps': 100,
        'max_context_length': 6000,
        'generation_batch_size': num_generations,
    })
    response = requests.post(url, headers=headers, json=data)
    return response


# "HuggingFaceTB/SmolLM2-135M-Instruct"
# "Qwen/Qwen2-0.5B-Instruct"
def run_grpo_step(
    model_name, batch, url=f"http://127.0.0.1:{arbor_port}/v1/fine_tuning/grpo/step"
):
    headers = {"Content-Type": "application/json"}
    data = {"model": model_name, "update_inference_model": True, "batch": batch}
    response = requests.post(url, headers=headers, json=data)
    return response


def update_model(
    url=f"http://127.0.0.1:{arbor_port}/v1/fine_tuning/grpo/update_model"
):
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers)
    return response


def checkpoint(
    checkpoint_name, url=f"http://127.0.0.1:{arbor_port}/v1/fine_tuning/grpo/checkpoint"
):
    headers = {"Content-Type": "application/json"}
    data = {"checkpoint_name": checkpoint_name}
    response = requests.post(url, headers=headers, json=data)
    return response


def terminate_grpo(url=f"http://127.0.0.1:{arbor_port}/v1/fine_tuning/grpo/terminate"):
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers)
    return response


def main():
    global grad_accum_steps, current_model, context_length
    def _reward_func(prompts, completions):

        return [
            -abs(20 - len(completion)) if completion is not None else -300
            for completion in completions
        ]

    dataset = load_dataset("trl-lib/tldr", split="train")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(current_model)
    # return len(tokenizer.apply_chat_template(messages))
    initialize_response = initialize_grpo(model=current_model)
    last_checkpoint = None

    for i in range(len(dataset)):
        inputs = dataset[i]
        input_messages = [{"role": "user", "content": inputs["prompt"]}]
        response = {"content": "Hello, world!", "role": "assistant"}
        token_len = len(tokenizer.apply_chat_template(input_messages + [response]))
        while token_len < context_length:
            response['content'] += "Hello, world!"
            token_len = len(tokenizer.apply_chat_template(input_messages + [response]))
        
        response['content'] = response['content'][:-1 * len("Hello, world!")]
        token_len = len(tokenizer.apply_chat_template(input_messages + [response]))
        print(token_len)

        completions = [response] * num_generations
        rewards = _reward_func(inputs["prompt"], [c["content"] for c in completions])
        print(rewards)

        batch = []
        for completion, reward in zip(completions, rewards):
            batch.append(
                {"messages": input_messages, "completion": completion, "reward": reward}
            )

        for i in range(grad_accum_steps):
            step_response = run_grpo_step(model_name=current_model, batch=batch)
            current_model = step_response.json()["current_model"]

        # if i % 42 == 0 and i > 0:
        update_response = update_model()
        current_model = update_response.json()["current_model"]

        # checkpoint_response = checkpoint(checkpoint_name=f"checkpoint_{i}")
        # last_checkpoint_name = checkpoint_response.json()["last_checkpoint_name"]
        # import pdb

        # pdb.set_trace()

    terminate_response = terminate_grpo()
    import pdb

    # pdb.set_trace()


if __name__ == "__main__":
    main()
