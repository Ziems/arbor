import random
from dataclasses import dataclass
from typing import List


@dataclass
class MockJob:
    id: int
    job_id: str
    name: str
    progress_percent: int
    runtime_seconds: int
    gpu_ids: List[int]
    gpu_utilization: int
    status: str


def generate_mock_jobs() -> List[MockJob]:
    """Generate 3-5 realistic RL training jobs"""
    job_configs = [
        ("bert-sentiment-opt", "job_001"),
        ("gpt-qa-optimization", "job_002"),
        ("llama-summarization", "job_003"),
        ("t5-translation-rl", "job_004"),
        ("roberta-classification", "job_005"),
    ]

    jobs = []
    for i, (name, job_id) in enumerate(job_configs[:3], 1):  # 3 mock jobs
        jobs.append(
            MockJob(
                id=i,
                job_id=job_id,
                name=name,
                progress_percent=random.randint(5, 95),
                runtime_seconds=random.randint(600, 28800),  # 10 min to 8 hours
                gpu_ids=_assign_mock_gpus(),
                gpu_utilization=random.randint(60, 95),
                status="Training",
            )
        )
    return jobs


def _assign_mock_gpus() -> List[int]:
    """Simulate GPU allocation patterns"""
    patterns = [[0], [1], [2], [3], [1, 2], [0, 3]]
    return random.choice(patterns)
