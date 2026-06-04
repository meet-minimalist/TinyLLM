import json
import random
import time

import torch
from datasets import load_dataset

from src.tinyllm.benchmarks.base import BaseBenchmark


class HellaSwagBenchmark(BaseBenchmark):
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
        model.eval()
        data = self._load_data()

        correct = 0
        total = 0

        for item in data:
            ctx = item["ctx"]
            endings = item["endings"]

            ctx_ids = tokenizer.encode(ctx, return_tensors="pt").to(device)
            ctx_len = ctx_ids.shape[1]

            scores = []
            for ending in endings:
                text = ctx + " " + ending
                input_ids = tokenizer.encode(text, return_tensors="pt").to(
                    device
                )
                logits = model(input_ids)
                shift_logits = logits[:, ctx_len - 1 : -1, :]
                shift_labels = input_ids[:, ctx_len:]
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.shape[-1]),
                    shift_labels.reshape(-1),
                    reduction="mean",
                )
                scores.append(-loss.item())

            if scores.index(max(scores)) == 0:
                correct += 1
            total += 1

        acc = correct / max(total, 1)
        elapsed = time.perf_counter() - t0
        return {"accuracy": acc, "num_samples": total, "time": elapsed}
