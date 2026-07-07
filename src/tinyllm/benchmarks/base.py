from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class BaseBenchmark(ABC):
    @abstractmethod
    def run(self, model, tokenizer, device):
        """Run benchmark evaluation.

        Args:
            model: nn.Module in eval mode.
            tokenizer: Tokenizer instance.
            device: torch device.

        Returns:
            Dict with at least {"accuracy": float}.
        """
        ...


def score_completions(
    model, tokenizer, device, context: str, completions: list
) -> list:
    """
    Score each completion by mean log-prob of its tokens given context.

    Encodes `context + completion` as a single sequence, then extracts the
    log-likelihood of just the completion tokens (everything after ctx_len).
    Higher score = model assigns more probability mass to that completion.

    Args:
        context: The shared prefix text (not scored).
        completions: List of completion strings to compare.

    Returns:
        List of float scores, one per completion (higher is better).
    """
    ctx_len = len(tokenizer.encode(context))
    scores = []
    for completion in completions:
        input_ids = tokenizer.encode(
            context + completion, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(input_ids)
        if ctx_len == 0:
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
        else:
            shift_logits = logits[:, ctx_len - 1 : -1, :]
            shift_labels = input_ids[:, ctx_len:]
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            reduction="mean",
        )
        scores.append(-loss.item())
    return scores
