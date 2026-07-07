import random
import time

import torch
from datasets import load_dataset

from src.tinyllm.benchmarks.base import BaseBenchmark, score_completions


class ARCEasyBenchmark(BaseBenchmark):
    """
    4-way multiple choice elementary science questions (ARC-Easy).

    Each question is formatted as "Question: ...\nAnswer:" and the model scores
    each answer choice as a completion. The choice with highest log-likelihood wins.

    Expected accuracy: 50M ≈ 33% (near random for 4-way), 3B ≈ 65%.
    Tracks factual knowledge accumulation — this benchmark requires more than
    just surface-level patterns and improves noticeably with model scale.
    """

    def __init__(self, num_samples: int = 200, seed: int = 42):
        self.num_samples = num_samples
        self.seed = seed
        self._dataset = None

    def _load_data(self):
        if self._dataset is not None:
            return self._dataset
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation")
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
            question = item["question"]
            choices_text = item["choices"]["text"]
            choices_label = item["choices"]["label"]
            answer_key = item["answerKey"]

            if answer_key not in choices_label:
                continue
            label = choices_label.index(answer_key)

            context = f"Question: {question}\nAnswer:"
            completions = [f" {c}" for c in choices_text]
            scores = score_completions(
                model, tokenizer, device, context, completions
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
