import random

import dspy
from datasets import load_dataset

import arbor
from arbor import ArborGRPO, ArborProvider

# Start Arbor server (starts in background)
arbor_server_info = arbor.init()

raw_dataset = load_dataset("Helsinki-NLP/opus_books", "en-fr")
raw_data = [
    dspy.Example(english=ex["translation"]["en"], french=ex["translation"]["fr"]).with_inputs("english")
    for ex in raw_dataset["train"]
][:2000]

random.Random(43).shuffle(raw_data)
trainset = raw_data[:1000]
testset = raw_data[1000:1100]

print(trainset[0])

unique_chars = dspy.Predict(f"english -> french")

# Get Arbor server info from init()
provider = ArborProvider()

student_lm_name = "Qwen/Qwen2.5-0.5B-Instruct"
# student_lm_name = "Qwen/Qwen2.5-7B-Instruct"
# student_lm_name = "Qwen/Qwen2.5-14B-Instruct"
# student_lm_name = "Qwen/Qwen2.5-32B-Instruct"
student_lm = dspy.LM(
    model=f"openai/arbor:{student_lm_name}",
    provider=provider,
    temperature=1.0,
    api_base=arbor_server_info["base_url"],
    api_key="arbor",
    max_tokens=2000,
)

student_unique_chars = unique_chars.deepcopy()
student_unique_chars.set_lm(student_lm)

def _unique_letter_reward(input, pred, trace=None) -> float:
    letters = [ch.lower() for ch in pred.french if ch.isalpha()]
    return float(len(set(letters)))


train_kwargs = {
    "per_device_train_batch_size": 8,
    "temperature": 1.0,
    "beta": 0.01,
    "learning_rate": 5e-5,
    "gradient_checkpointing": True,
    "gradient_accumulation_steps": 3,
    "bf16": True,
    "lr_scheduler_type": "constant_with_warmup",
    "warmup_steps": 100,
    # "max_prompt_length": None,
    # "max_completion_length": 1024,
    "report_to": "wandb",  # 'wandb'
    "log_completions": False,
    "max_context_length": None,
    "max_steps": 1000,
    "logging_steps": 1,
    "lora_config": {
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "lora_r": 8,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    },
}

compiler = ArborGRPO(
    metric=_unique_letter_reward,
    num_dspy_examples_per_grpo_step=1,
    num_rollouts_per_grpo_step=24,
    exclude_demos=True,
    num_train_steps=20,
    num_threads=1,
    use_train_as_val=False,
    num_steps_for_val=10,
    train_kwargs=train_kwargs,
)

classify_ft = compiler.compile(
    student=student_unique_chars,
    trainset=trainset,
    valset=testset,
)

evaluate = dspy.Evaluate(
    devset=testset,
    metric=_unique_letter_reward,
    display_progress=True,
    display_table=5,
    num_threads=16,
)
