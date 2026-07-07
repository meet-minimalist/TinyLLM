import random
import time

import torch
from datasets import load_dataset

from src.tinyllm.benchmarks.base import BaseBenchmark, score_completions


class HellaSwagBenchmark(BaseBenchmark):
    """
    4-way completion scoring benchmark for commonsense NLI.
    Correct answer index is stored in item["label"] (string "0"-"3").
    """

    def __init__(self, num_samples: int = 200, seed: int = 42):
        self.num_samples = num_samples
        self.seed = seed
        self._dataset = None

    def _load_data(self):
        if self._dataset is not None:
            return self._dataset
        ds = load_dataset(
            "Rowan/hellaswag", split="validation", trust_remote_code=True
        )
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
            ctx = item["ctx"]
            endings = item["endings"]
            label = int(item["label"])

            completions = [" " + e for e in endings]
            scores = score_completions(
                model, tokenizer, device, ctx, completions
            )

            if scores.index(max(scores)) == label:
                correct += 1
            total += 1

        acc = correct / max(total, 1)
        elapsed = time.perf_counter() - t0
        return {"accuracy": acc, "num_samples": total, "time": elapsed}
