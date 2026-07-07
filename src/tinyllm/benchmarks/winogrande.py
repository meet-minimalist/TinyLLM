import random
import time

import torch
from datasets import load_dataset

from src.tinyllm.benchmarks.base import BaseBenchmark, score_completions


class WinograndeBenchmark(BaseBenchmark):
    """
    2-way pronoun disambiguation benchmark (commonsense reasoning).

    Each example has a sentence with an underscore placeholder and two candidate
    words. The sentence is split at the underscore; only the completion (chosen
    word + trailing text) is scored against the shared prefix. This isolates the
    part that differs between options and avoids length bias.

    Expected accuracy: 50M ≈ 51% (near random), 3B ≈ 64%.
    Useful for tracking when commonsense coreference resolution emerges.
    """

    def __init__(self, num_samples: int = 300, seed: int = 42):
        self.num_samples = num_samples
        self.seed = seed
        self._dataset = None

    def _load_data(self):
        if self._dataset is not None:
            return self._dataset
        ds = load_dataset(
            "allenai/winogrande", "winogrande_xl", split="validation"
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
            sentence = item["sentence"]
            options = [item["option1"], item["option2"]]
            label = int(item["answer"]) - 1  # "1" -> 0, "2" -> 1

            blank_idx = sentence.find("_")
            if blank_idx == -1:
                continue

            pre = sentence[:blank_idx]
            post = sentence[blank_idx + 1 :]

            # Score: option + remaining sentence given the prefix
            completions = [opt + post for opt in options]
            scores = score_completions(
                model, tokenizer, device, pre, completions
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
