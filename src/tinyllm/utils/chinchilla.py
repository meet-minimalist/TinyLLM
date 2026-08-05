"""
Chinchilla-optimal token budget helpers (Hoffmann et al., 2022).

Chinchilla's compute-optimal finding is that a model with ``N`` parameters
should be trained on roughly ``D ≈ 20 · N`` tokens. Given how many tokens each
training step processes, this converts that token budget into a concrete number
of steps — so you know up front how long to train and when training will stop.

Note on N: we count *all* model parameters (including the tied token embedding).
The strict Chinchilla ``N`` is often the non-embedding parameter count; for the
small, embedding-heavy models here that would give a smaller budget. Tune
``tokens_per_param`` if you want to match a particular convention or intentionally
over-/under-train (e.g. 2× Chinchilla ≈ 40).
"""

import math

DEFAULT_TOKENS_PER_PARAM = 20.0


def tokens_per_step(train_config) -> int:
    """Tokens processed per training step (one micro-batch).

    global_step increments once per micro-batch, and num_training_steps is
    compared against it, so a "step" is one batch regardless of grad accumulation.
    """
    mode = train_config.get("mode", "varlen_packed")
    if mode == "varlen_packed":
        return int(train_config.get("packed_tokens", 8192))
    # fixed_batch mode: batch_size × sequence length
    batch_size = int(train_config.get("batch_size", 1))
    seq_len = int(
        train_config.get("seq_len", train_config.get("max_seq_len", 2048))
    )
    return batch_size * seq_len


def chinchilla_token_budget(
    n_params: int, tokens_per_param: float = DEFAULT_TOKENS_PER_PARAM
) -> int:
    """Chinchilla-optimal token budget: ``tokens_per_param · N``."""
    return int(tokens_per_param * n_params)


def chinchilla_num_steps(
    n_params: int,
    tokens_per_step_: int,
    tokens_per_param: float = DEFAULT_TOKENS_PER_PARAM,
) -> tuple[int, int]:
    """Return ``(num_steps, target_tokens)`` for a Chinchilla-optimal run.

    ``num_steps = ceil(tokens_per_param · N / tokens_per_step)``.
    """
    target_tokens = chinchilla_token_budget(n_params, tokens_per_param)
    steps = max(1, math.ceil(target_tokens / max(1, int(tokens_per_step_))))
    return steps, target_tokens
