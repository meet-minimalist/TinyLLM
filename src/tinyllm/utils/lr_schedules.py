"""
LR schedules, registered in ``LR_SCHEDULER_REGISTRY``.

Every entry has the same signature::

    build(optimizer, num_training_steps, num_warmup_steps, sched_cfg) -> LRScheduler

``sched_cfg`` is the ``lr_schedule`` block of the training config. Step values
in it that are floats in (0, 1) are fractions of ``num_training_steps``.

All schedules except ``power`` come from ``transformers.optimization``. All
are ``LambdaLR`` multipliers on each param group's base LR, so they work with
the Muon+AdamW hybrid (two base LRs) and resume via ``load_state_dict``.

To add a schedule: write a function with the signature above and decorate it
with ``@LR_SCHEDULER_REGISTRY.register("<name>")``; then set
``lr_scheduler_type: "<name>"`` in the training config.
"""

from torch.optim.lr_scheduler import LambdaLR

from src.tinyllm.factory.registry import LR_SCHEDULER_REGISTRY


def _steps(value, num_training_steps: int) -> int:
    """A float in (0, 1) is a fraction of ``num_training_steps``."""
    if isinstance(value, float) and 0 < value < 1:
        return int(value * num_training_steps)
    return int(value)


# --- transformers schedules ---------------------------------------------------
@LR_SCHEDULER_REGISTRY.register("cosine")
def cosine(optimizer, num_training_steps, num_warmup_steps, sched_cfg):
    from transformers.optimization import get_cosine_schedule_with_warmup

    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )


@LR_SCHEDULER_REGISTRY.register("linear")
def linear(optimizer, num_training_steps, num_warmup_steps, sched_cfg):
    from transformers.optimization import get_linear_schedule_with_warmup

    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )


@LR_SCHEDULER_REGISTRY.register("constant")
def constant(optimizer, num_training_steps, num_warmup_steps, sched_cfg):
    from transformers.optimization import get_constant_schedule

    return get_constant_schedule(optimizer)


@LR_SCHEDULER_REGISTRY.register("constant_warmup")
def constant_warmup(optimizer, num_training_steps, num_warmup_steps, sched_cfg):
    from transformers.optimization import get_constant_schedule_with_warmup

    return get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps
    )


@LR_SCHEDULER_REGISTRY.register("inverse_sqrt")
def inverse_sqrt(optimizer, num_training_steps, num_warmup_steps, sched_cfg):
    from transformers.optimization import get_inverse_sqrt_schedule

    return get_inverse_sqrt_schedule(
        optimizer, num_warmup_steps=num_warmup_steps
    )


@LR_SCHEDULER_REGISTRY.register("wsd")
def wsd(optimizer, num_training_steps, num_warmup_steps, sched_cfg):
    """Warmup -> stable -> decay (MiniCPM, arXiv:2404.06395)."""
    from transformers.optimization import get_wsd_schedule

    return get_wsd_schedule(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_decay_steps=_steps(
            sched_cfg.get("decay_steps", 0.2), num_training_steps
        ),
        num_training_steps=num_training_steps,
        decay_type=sched_cfg.get("decay_type", "linear"),
        min_lr_ratio=float(sched_cfg.get("min_lr_ratio", 0.0)),
    )


# --- power: not in transformers -----------------------------------------------
def power_lambda(
    num_training_steps: int,
    num_warmup_steps: int,
    power_start_step: int,
    exponent: float,
    decay_steps: int,
    min_lr_ratio: float = 0.0,
):
    """warmup -> constant -> ``(s / s0) ** -exponent`` -> linear decay."""
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


@LR_SCHEDULER_REGISTRY.register("power")
def power(optimizer, num_training_steps, num_warmup_steps, sched_cfg):
    """IBM Power scheduler (arXiv:2408.13359), the schedule Rigel uses.

    The power-law part does not depend on the total step count, so the run
    length can grow without re-tuning.
    """
    start = _steps(
        sched_cfg.get("power_start_step", num_warmup_steps), num_training_steps
    )
    fn = power_lambda(
        num_training_steps,
        num_warmup_steps,
        power_start_step=max(1, start, num_warmup_steps),
        exponent=float(sched_cfg.get("exponent", 0.51)),
        decay_steps=_steps(
            sched_cfg.get("decay_steps", 0.2), num_training_steps
        ),
        min_lr_ratio=float(sched_cfg.get("min_lr_ratio", 0.0)),
    )
    return LambdaLR(optimizer, fn)
