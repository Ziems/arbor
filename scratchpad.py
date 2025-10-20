import arbor
import dspy
import random
from dotenv import load_dotenv
import os

load_dotenv()
# Start Arbor server (auto-detects GPUs, starts in background)
arbor.init()

# Sample classification data
data = [
    dspy.Example(text="I want to transfer money", label="transfer").with_inputs("text"),
    dspy.Example(text="What is my balance?", label="balance").with_inputs("text"),
    dspy.Example(text="I lost my credit card", label="card_issues").with_inputs("text"),
    # ... more examples
]

# Split into train/validation
random.Random(42).shuffle(data)
trainset, valset = data[:6], data[6:]

# Define classification task
CLASSES = ["transfer", "balance", "card_issues", "pin_change"]
classify = dspy.ChainOfThought(f"text -> label: Literal{CLASSES}")

# Set up DSPy with Arbor backend
from arbor import ArborProvider

provider = ArborProvider()
student_lm = dspy.LM(
    model="openai/arbor:Qwen/Qwen2-0.5B-Instruct",
    provider=provider,
    api_base="http://127.0.0.1:7453/v1/",
    api_key="arbor",
    hf_token=os.getenv("HF_TOKEN"),
)

student_classify = classify.deepcopy()
student_classify.set_lm(student_lm)

# Optimize with Arbor's GRPO trainer (requires 2+ GPUs)
from arbor import ArborGRPO

compiler = ArborGRPO(metric=lambda x, y: x.label == y.label, exclude_demos=True)

# Run optimization
optimized_classify = compiler.compile(
    student=student_classify, trainset=trainset, valset=valset
)
# print(student_classify(text="I want to transfer money"))


# Your classifier is now optimized with RL! 🎉
