"""
Power LR schedule (IBM Power scheduler, arXiv:2408.13359; the schedule Rigel uses).

warmup -> constant -> power-law decay ``(s / s0) ** -exponent`` -> linear decay
to ``min_lr_ratio``. The power-law part does not depend on the total step
count, so the run length can grow without re-tuning.

``transformers`` has no equivalent; every other schedule (cosine, linear, wsd,
...) comes from ``transformers.optimization`` via ``lr_scheduler_factory``.
It is a ``LambdaLR`` multiplier on each param group's base LR, so it works
with the Muon+AdamW hybrid (two base LRs) and resumes via ``load_state_dict``.
"""

from torch.optim.lr_scheduler import LambdaLR


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


def get_power_schedule(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    power_start_step: int,
    exponent: float = 0.51,
    num_decay_steps: int = 0,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """Same calling convention as the ``transformers.optimization`` schedules."""
    fn = power_lambda(
        num_training_steps,
        num_warmup_steps,
        power_start_step=max(1, power_start_step, num_warmup_steps),
        exponent=exponent,
        decay_steps=num_decay_steps,
        min_lr_ratio=min_lr_ratio,
    )
    return LambdaLR(optimizer, fn)
