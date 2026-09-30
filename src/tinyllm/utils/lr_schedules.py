"""
LR schedules for long, resumable pretraining runs.

All schedules are ``LambdaLR`` multipliers on each param group's base LR, so
they work unchanged with the Muon+AdamW hybrid (two groups, two base LRs) and
their position is restored by ``LambdaLR.load_state_dict`` on resume.

- ``wsd``: warmup -> stable -> decay (MiniCPM, arXiv:2404.06395). The stable
  phase has no fixed end, so a decay (and mid-training data) can be branched off
  any stable checkpoint.
- ``power``: warmup -> constant -> power-law decay ``(s / s0) ** -exponent`` ->
  linear decay to ``min_lr_ratio`` (IBM Power scheduler, arXiv:2408.13359; the
  schedule Rigel uses). The power-law part does not depend on the total step
  count, so the run length can grow without re-tuning.
"""

import math

from torch.optim.lr_scheduler import LambdaLR


def _resolve_steps(value, total_steps: int) -> int:
    """A float in (0, 1) is a fraction of ``total_steps``; anything else is a count."""
    if isinstance(value, float) and 0 < value < 1:
        return int(value * total_steps)
    return int(value)


def _decay_shape(progress: float, shape: str) -> float:
    """Multiplier in [0, 1] that goes from 1 (progress=0) to 0 (progress=1)."""
    progress = min(max(progress, 0.0), 1.0)
    if shape == "linear":
        return 1.0 - progress
    if shape == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    if shape == "sqrt":  # "1 - sqrt" cooldown (Hägele et al. 2024)
        return 1.0 - math.sqrt(progress)
    raise ValueError(f"Unknown decay shape: {shape}")


def wsd_lambda(
    num_training_steps: int,
    num_warmup_steps: int,
    decay_steps: int,
    decay_shape: str = "linear",
    min_lr_ratio: float = 0.0,
):
    decay_start = num_training_steps - decay_steps

    def fn(step: int) -> float:
        if step < num_warmup_steps:
            return (step + 1) / max(1, num_warmup_steps)
        if step < decay_start:
            return 1.0
        progress = (step - decay_start) / max(1, decay_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * _decay_shape(
            progress, decay_shape
        )

    return fn


def power_lambda(
    num_training_steps: int,
    num_warmup_steps: int,
    power_start_step: int,
    exponent: float,
    decay_steps: int,
    min_lr_ratio: float = 0.0,
):
    decay_start = num_training_steps - decay_steps

    def power(step: int) -> float:
        if step <= power_start_step:
            return 1.0
        return (step / power_start_step) ** -exponent

    def fn(step: int) -> float:
        if step < num_warmup_steps:
            return (step + 1) / max(1, num_warmup_steps)
        if step < decay_start:
            return power(step)
        # Linear decay from wherever the power law has reached.
        start = power(decay_start)
        progress = min((step - decay_start) / max(1, decay_steps), 1.0)
        end = min_lr_ratio * start
        return start + (end - start) * progress

    return fn


def build_lambda_schedule(
    name: str,
    optimizer,
    num_training_steps: int,
    num_warmup_steps: int,
    sched_cfg: dict,
) -> LambdaLR:
    """Build ``wsd`` or ``power``. ``sched_cfg`` is the ``lr_schedule`` config block."""
    decay_steps = _resolve_steps(
        sched_cfg.get("decay_steps", 0.2), num_training_steps
    )
    min_lr_ratio = float(sched_cfg.get("min_lr_ratio", 0.0))
    if name == "wsd":
        fn = wsd_lambda(
            num_training_steps,
            num_warmup_steps,
            decay_steps,
            decay_shape=sched_cfg.get("decay_shape", "linear"),
            min_lr_ratio=min_lr_ratio,
        )
    elif name == "power":
        power_start = _resolve_steps(
            sched_cfg.get("power_start_step", num_warmup_steps),
            num_training_steps,
        )
        fn = power_lambda(
            num_training_steps,
            num_warmup_steps,
            power_start_step=max(1, power_start, num_warmup_steps),
            exponent=float(sched_cfg.get("exponent", 0.51)),
            decay_steps=decay_steps,
            min_lr_ratio=min_lr_ratio,
        )
    else:
        raise ValueError(f"Unknown lambda schedule: {name}")
    return LambdaLR(optimizer, fn)
