import random

import dspy
from datasets import load_dataset
from dspy.datasets import DataLoader

import arbor

# Start Arbor server (starts in background)
arbor_server_info = arbor.init()

raw_data = [
    dspy.Example({"prompt": x['prompt'][0]['content']}).with_inputs("prompt")
    for x in DataLoader().from_huggingface("trl-lib/ultrafeedback-prompt", split="train")
][:2000]

random.Random(43).shuffle(raw_data)
trainset = raw_data[:1000]

print(trainset[0])

unique_chars = dspy.Predict(f"prompt -> output")

from dspy.clients.lm_local_arbor import ArborProvider
# Get Arbor server info from init()
provider = ArborProvider()

student_lm_name = "Qwen/Qwen2.5-1.5B-Instruct"
# student_lm_name = "Qwen/Qwen2.5-7B-Instruct"
# student_lm_name = "Qwen/Qwen2.5-14B-Instruct"
# student_lm_name = "Qwen/Qwen2.5-32B-Instruct"
student_lm = dspy.LM(
    model=f"openai/arbor:{student_lm_name}",
    provider=provider,
    temperature=1.0,
    api_base=arbor_server_info["base_url"],
    api_key="arbor",
    # max_tokens=4000,
)

student_unique_chars = unique_chars.deepcopy()
student_unique_chars.set_lm(student_lm)

def _unique_letter_reward(input, pred, trace=None) -> float:
    letters = [ch.lower() for ch in pred.output if ch.isalpha()]
    return float(len(set(letters)))


from dspy.teleprompt.grpo import GRPO

train_kwargs = {
    "per_device_train_batch_size": 4,
    "temperature": 1.0,
    "beta": 0.02,
    "learning_rate": 5e-6,
    "gradient_checkpointing": True,
    "gradient_accumulation_steps": 8,
    "bf16": True,
    "lr_scheduler_type": "constant_with_warmup",
    # "max_prompt_length": None,
    # "max_completion_length": 1024,
    # "lora": True,
    "report_to": "none",  # 'wandb'
    "log_completions": False,
    "max_context_length": None,
    "max_steps": 1000,
    "max_model_len": 4096,
}

compiler = GRPO(
    metric=_unique_letter_reward,
    num_dspy_examples_per_grpo_step=1,
    num_rollouts_per_grpo_step=4,
    exclude_demos=True,
    num_train_steps=20,
    num_threads=1,
    use_train_as_val=False,
    num_steps_for_val=10,
    train_kwargs=train_kwargs,
)

try:
    classify_ft = compiler.compile(
        student=student_unique_chars,
        trainset=trainset,
        )
except Exception as e:
    print(dspy.inspect_history())
    import pdb
    pdb.set_trace()
    raise e

# evaluate = dspy.Evaluate(
#     devset=testset,
#     metric=metric,
#     display_progress=True,
#     display_table=5,
#     num_threads=16,
# )

# import pdb

# pdb.set_trace()
