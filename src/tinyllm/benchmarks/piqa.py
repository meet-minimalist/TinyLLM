import random
import time

import torch
from datasets import load_dataset

from src.tinyllm.benchmarks.base import BaseBenchmark, score_completions


class PIQABenchmark(BaseBenchmark):
    """
    2-way physical intuition QA benchmark.

    Each example has a goal and two solutions; the model scores each solution
    as a completion to the goal text. Physical intuition requires grounding in
    everyday cause-and-effect that a model learns from naturalistic text.

    Expected accuracy: 50M ≈ 60%, 3B ≈ 77%.
    Because random baseline is 50% and 50M models already beat it noticeably,
    this benchmark shows improvement early in training compared to ARC-Easy.
    """

    def __init__(self, num_samples: int = 300, seed: int = 42):
        self.num_samples = num_samples
        self.seed = seed
        self._dataset = None

    def _load_data(self):
        if self._dataset is not None:
            return self._dataset
        # The canonical HF "piqa" dataset is script-based and no longer loads on
        # datasets>=4 ("Dataset scripts are no longer supported"). lighteval/piqa
        # is a parquet mirror with the same goal/sol1/sol2/label fields.
        ds = load_dataset("lighteval/piqa", split="validation")
        rng = random.Random(self.seed)
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        self._dataset = [ds[i] for i in indices[: self.num_samples]]
        return self._dataset

    @torch.no_grad()
    def run(self, model, tokenizer, device):
        t0 = time.perf_counter()
        data = self._load_data()

        correct = 0
        total = 0

        for item in data:
            goal = item["goal"]
            solutions = [item["sol1"], item["sol2"]]
            label = item["label"]  # 0 or 1

            completions = [" " + s for s in solutions]
            scores = score_completions(
                model, tokenizer, device, goal, completions
            )

            if scores.index(max(scores)) == label:
                correct += 1
            total += 1

        elapsed = time.perf_counter() - t0
        return {
            "accuracy": correct / max(total, 1),
            "num_samples": total,
            "time": elapsed,
        }
