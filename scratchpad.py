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
    dspy.Example(text="How can I change my PIN?", label="pin_change").with_inputs(
        "text"
    ),
    dspy.Example(text="Show me my last transactions", label="balance").with_inputs(
        "text"
    ),
    dspy.Example(
        text="There's an unauthorized transaction on my account", label="card_issues"
    ).with_inputs("text"),
    dspy.Example(text="Reset my password", label="card_issues").with_inputs("text"),
    dspy.Example(text="Deposit $100 to my account", label="transfer").with_inputs(
        "text"
    ),
    dspy.Example(text="Update my mailing address", label="card_issues").with_inputs(
        "text"
    ),
    dspy.Example(text="I forgot my PIN", label="pin_change").with_inputs("text"),
    dspy.Example(text="Transfer $50 to John", label="transfer").with_inputs("text"),
    dspy.Example(text="How much can I withdraw today?", label="balance").with_inputs(
        "text"
    ),
]

# Split into train/validation
random.Random(42).shuffle(data)
trainset, valset = data[:6], data[6:]

# Define classification task
CLASSES = ["transfer", "balance", "card_issues", "pin_change"]
classify = dspy.Predict("text -> label")

# Set up DSPy with Arbor backend
from arbor import ArborProvider

provider = ArborProvider()
student_lm = dspy.LM(
    model="openai/arbor:tytodd/Qwen2-0.5B-Instruct",
    provider=provider,
    api_base="http://127.0.0.1:7453/v1/",
    api_key="arbor",
    hf_token=os.getenv("HF_TOKEN"),
)

student_classify = classify.deepcopy()
student_classify.set_lm(student_lm)

# Optimize with Arbor's GRPO trainer (requires 2+ GPUs)
from arbor import ArborGRPO, ArborHFConfig

compiler = ArborGRPO(
    metric=lambda x, y: x.label == y.label,
    exclude_demos=True,
    num_rollouts_per_grpo_step=4,
    hf_config=ArborHFConfig(
        hub_model_id="tytodd/arbor-test-2",
        hub_token=os.getenv("HF_TOKEN"),
        push_frequency="final_checkpoint",
    ),
)

# Run optimization
optimized_classify = compiler.compile(
    student=student_classify, trainset=trainset, valset=valset
)
# print(student_classify(text="I want to transfer money"))


# Your classifier is now optimized with RL! 🎉
