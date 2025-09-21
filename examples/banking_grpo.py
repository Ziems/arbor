import random

import dspy
from datasets import load_dataset
from dspy.clients.utils_finetune import MultiGPUConfig
from dspy.datasets import DataLoader

import arbor

# Start Arbor server (starts in background)
arbor_server_info = arbor.init()

CLASSES = (
    load_dataset("PolyAI/banking77", split="train", trust_remote_code=True)
    .features["label"]
    .names
)
kwargs = dict(
    fields=("text", "label"),
    input_keys=("text",),
    split="train",
    trust_remote_code=True,
)

TOP_CLASSES = CLASSES[:]

raw_data = [
    dspy.Example(x, label=CLASSES[x.label]).with_inputs("text")
    for x in DataLoader().from_huggingface(dataset_name="PolyAI/banking77", **kwargs)
    if CLASSES[x.label] in TOP_CLASSES
][:2000]

random.Random(42).shuffle(raw_data)
print(len(TOP_CLASSES), TOP_CLASSES)
trainset = raw_data[:1000]
devset = raw_data[1000:1100]
testset = raw_data[1100:1200]
assert len(devset) > 30
assert len(testset) > 30

# print(trainset[0])

classify = dspy.ChainOfThought(f"text -> label: Literal{TOP_CLASSES}")

from dspy.clients.lm_local_arbor import ArborProvider

# Get Arbor server info from init()
provider = ArborProvider()

#student_lm_name = "Qwen/Qwen2.5-1.5B-Instruct"
student_lm_name = "Qwen/Qwen2.5-7B-Instruct"
#student_lm_name = "Qwen/Qwen2.5-14B-Instruct"
#student_lm_name = "Qwen/Qwen2.5-32B-Instruct"
student_lm = dspy.LM(
    model=f"openai/arbor:{student_lm_name}",
    provider=provider,
    temperature=1.0,
    api_base=arbor_server_info["base_url"],
    api_key="arbor",
)

student_classify = classify.deepcopy()
student_classify.set_lm(student_lm)

metric = lambda x, y, trace=None: x.label == y.label

from dspy.teleprompt.grpo import GRPO

train_kwargs = {
    "per_device_train_batch_size": 1,
    "temperature": 1.0, 
    "beta": 0.02,
    "learning_rate": 5e-6,
    "gradient_checkpointing": True,
    "gradient_accumulation_steps": 8,
    "bf16": True,
    "lr_scheduler_type": "constant_with_warmup",
    "max_prompt_length": None,
    "max_completion_length": None,
    "lora": True,
    "report_to": "none",  # 'wandb'
    "log_completions": False,
}

compiler = GRPO(
    metric=metric,
    num_dspy_examples_per_grpo_step=4,
    num_rollouts_per_grpo_step=4,
    exclude_demos=True,
    num_train_steps=20,
    num_threads=1,
    use_train_as_val=False,
    num_steps_for_val=10,
    train_kwargs=train_kwargs,
    gpu_config=MultiGPUConfig(num_inference_gpus=1, num_training_gpus=1),
)

classify_ft = compiler.compile(
    student=student_classify,
    trainset=trainset,
    valset=devset,
)

import pdb; pdb.set_trace()

evaluate = dspy.Evaluate(
    devset=testset,
    metric=metric,
    display_progress=True,
    display_table=5,
    num_threads=16,
)

import pdb; pdb.set_trace()
