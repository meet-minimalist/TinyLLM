import math
import random
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset

from src.tinyllm.benchmarks.base import BaseBenchmark


class LambadaBenchmark(BaseBenchmark):
    """
    Last-word prediction accuracy on the LAMBADA OpenAI test set.

    The model is given a passage with the last word removed and must predict it.
    Accuracy = fraction of examples where greedy decoding recovers all tokens
    of the last word exactly. Also reports perplexity on the last-word tokens.

    This is the most direct signal for long-range language modelling quality
    and scales cleanly from 50M → 3B: 50M ≈ 25%, GPT-3 175B ≈ 76%.
    """

    def __init__(self, num_samples: int = 500, seed: int = 42):
        self.num_samples = num_samples
        self.seed = seed
        self._dataset = None

    def _load_data(self):
        if self._dataset is not None:
            return self._dataset
        ds = load_dataset("EleutherAI/lambada_openai", split="test")
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
        total_loss = 0.0
        n = 0

        for item in data:
            text = item["text"].strip()
            last_space = text.rfind(" ")
            if last_space == -1:
                continue

            context = text[:last_space]
            last_word = " " + text[last_space + 1 :]

            ctx_len = len(tokenizer.encode(context))
            full_ids = tokenizer.encode(
                context + last_word, return_tensors="pt"
            ).to(device)

            logits = model(full_ids)

            target_logits = logits[0, ctx_len - 1 : -1, :]
            target_ids = full_ids[0, ctx_len:]

            if target_ids.numel() == 0:
                continue

            loss = F.cross_entropy(target_logits, target_ids, reduction="mean")
            total_loss += loss.item()

            predictions = target_logits.argmax(-1)
            if (predictions == target_ids).all():
                correct += 1
            n += 1

        elapsed = time.perf_counter() - t0
        return {
            "accuracy": correct / max(n, 1),
            "perplexity": math.exp(total_loss / max(n, 1)),
            "num_samples": n,
            "time": elapsed,
        }
