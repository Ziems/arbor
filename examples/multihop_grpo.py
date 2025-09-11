import random

import dspy
from datasets import load_dataset
from dspy.clients.utils_finetune import MultiGPUConfig
from dspy.datasets import DataLoader

import arbor

arbor.init()

from dspy.clients.lm_local_arbor import ArborProvider

# Get Arbor server info from init()
server_info = arbor.status()
provider = ArborProvider()

student_lm_name = "Qwen/Qwen2.5-1.5B-Instruct"
student_lm = dspy.LM(
    model=f"openai/arbor:{student_lm_name}",
    provider=provider,
    temperature=1.0,
    # api_base='http://localhost:7453/v1/',
    api_base=server_info["base_url"],
    api_key="arbor",
)


########################################################

from dspy.utils import download

download("https://huggingface.co/dspy/cache/resolve/main/wiki.abstracts.2017.tar.gz")
#!tar -xzvf wiki.abstracts.2017.tar.gz

########################################################

import ujson

corpus = []

with open("wiki.abstracts.2017.jsonl") as f:
    for line in f:
        line = ujson.loads(line)
        corpus.append(f"{line['title']} | {' '.join(line['text'])}")

len(corpus)

########################################################

import bm25s
import Stemmer

stemmer = Stemmer.Stemmer("english")
corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)

retriever = bm25s.BM25(k1=0.9, b=0.4)
retriever.index(corpus_tokens)

########################################################

import random

from dspy.datasets import DataLoader

kwargs = dict(
    fields=("claim", "supporting_facts", "hpqa_id", "num_hops"), input_keys=("claim",)
)
hover = DataLoader().from_huggingface(
    dataset_name="hover-nlp/hover", split="train", trust_remote_code=True, **kwargs
)

hpqa_ids = set()
hover = [
    dspy.Example(
        claim=x.claim, titles=list(set([y["key"] for y in x.supporting_facts]))
    ).with_inputs("claim")
    for x in hover
    if x["num_hops"] == 3
    and x["hpqa_id"] not in hpqa_ids
    and not hpqa_ids.add(x["hpqa_id"])
]

random.Random(0).shuffle(hover)
trainset, devset, testset = hover[:200], hover[200:500], hover[650:]

########################################################


example = trainset[0]

print("Claim:", example.claim)
print("Pages that must be retrieved:", example.titles)

########################################################


def search(query: str, k: int) -> list[str]:
    tokens = bm25s.tokenize(query, stopwords="en", stemmer=stemmer, show_progress=False)
    results, scores = retriever.retrieve(tokens, k=k, n_threads=1, show_progress=False)
    run = {corpus[doc]: float(score) for doc, score in zip(results[0], scores[0])}
    return run


########################################################


class Hop(dspy.Module):
    def __init__(self, num_docs=10, num_hops=4):
        self.num_docs, self.num_hops = num_docs, num_hops
        self.generate_query = dspy.ChainOfThought("claim, notes -> query")
        self.append_notes = dspy.ChainOfThought(
            "claim, notes, context -> new_notes: list[str], titles: list[str]"
        )

    def forward(self, claim: str) -> list[str]:
        notes = []
        titles = []

        for _ in range(self.num_hops):
            query = self.generate_query(claim=claim, notes=notes).query
            context = search(query, k=self.num_docs)
            prediction = self.append_notes(claim=claim, notes=notes, context=context)
            notes.extend(prediction.new_notes)
            titles.extend(prediction.titles)

        return dspy.Prediction(notes=notes, titles=list(set(titles)))


########################################################


def top5_recall(example, pred, trace=None):
    gold_titles = example.titles
    recall = sum(x in pred.titles[:5] for x in gold_titles) / len(gold_titles)

    # If we're "bootstrapping" for optimization, return True if and only if the recall is perfect.
    if trace is not None:
        return recall >= 1.0

    # If we're just doing inference, just measure the recall.
    return recall


evaluate = dspy.Evaluate(
    devset=devset,
    metric=top5_recall,
    num_threads=16,
    display_progress=True,
    display_table=5,
)
baseline_hop = Hop().deepcopy()
baseline_hop.set_lm(student_lm)
evaluate(baseline_hop)

########################################################

from dspy.teleprompt.grpo import GRPO

train_kwargs = {
    "per_device_train_batch_size": 4,
    "temperature": 1.0,
    "beta": 0.02,
    "learning_rate": 1e-5,
    "gradient_checkpointing": False,
    "bf16": True,
    "lr_scheduler_type": "constant_with_warmup",
    "max_prompt_length": None,
    "max_completion_length": None,
    "lora": False,
    "report_to": "none",  # 'wandb'
    "log_completions": False,
}

compiler = GRPO(
    metric=metric,
    multitask=True,
    num_dspy_examples_per_grpo_step=4,
    num_rollouts_per_grpo_step=4,
    exclude_demos=True,
    num_train_steps=50,
    num_threads=8,
    use_train_as_val=False,
    num_steps_for_val=10,
    train_kwargs=train_kwargs,
    gpu_config=MultiGPUConfig(num_inference_gpus=1, num_training_gpus=1),
)

optimized_hop = compiler.compile(
    student=baseline_hop,
    trainset=trainset,
    valset=valset,
)

evaluate(optimized_hop)
