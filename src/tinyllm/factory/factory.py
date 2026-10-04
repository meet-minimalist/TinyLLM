from box import Box

from src.tinyllm.factory.registry import MODEL_REGISTRY
from src.tinyllm.logger.logger_utils import logger


def model_factory(model_config: Box):
    if not hasattr(model_config, "model_type"):
        raise ValueError("model_type must be specified in model_config.")

    import src.tinyllm.models

    return MODEL_REGISTRY.get(model_config.model_type)(model_config)


def optimizer_factory(model, train_config):
    opt_type = train_config.get("optimizer", {}).get("type", "adamw")

    if opt_type == "muon_adamw":
        from src.tinyllm.optimizer.muon import build_muon_adamw_optimizer

        opt_cfg = train_config.get("optimizer", {})
        use_8bit = opt_cfg.get("adamw_8bit", False)
        logger.info(
            f"Using Muon + {'8-bit ' if use_8bit else ''}AdamW hybrid optimizer"
        )
        return build_muon_adamw_optimizer(
            model,
            muon_lr=opt_cfg.get("muon_lr", 0.0002),
            adamw_lr=opt_cfg.get(
                "adamw_lr", train_config.get("init_lr", 0.001)
            ),
            weight_decay=opt_cfg.get("weight_decay", 0.1),
            momentum=opt_cfg.get("muon_momentum", 0.95),
            adamw_betas=opt_cfg.get("adamw_betas", (0.9, 0.95)),
            use_8bit=use_8bit,
        )

    import torch

    logger.info("Using AdamW optimizer")
    return torch.optim.AdamW(
        model.parameters(),
        lr=train_config.get("init_lr", 3e-4),
        weight_decay=0.1,
        fused=True,
    )


def lr_scheduler_factory(
    scheduler_name: str,
    optimizer,
    num_training_steps: int,
    num_warmup_steps: int = 0,
    sched_cfg: dict | None = None,
):
    sched_cfg = sched_cfg or {}

    def _steps(value) -> int:
        """A float in (0, 1) is a fraction of num_training_steps."""
        if isinstance(value, float) and 0 < value < 1:
            return int(value * num_training_steps)
        return int(value)

    num_decay_steps = _steps(sched_cfg.get("decay_steps", 0.2))
    min_lr_ratio = float(sched_cfg.get("min_lr_ratio", 0.0))

    if scheduler_name == "wsd":
        from transformers.optimization import get_wsd_schedule

        return get_wsd_schedule(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_decay_steps=num_decay_steps,
            num_training_steps=num_training_steps,
            decay_type=sched_cfg.get("decay_type", "linear"),
            min_lr_ratio=min_lr_ratio,
        )
    if scheduler_name == "power":
        from src.tinyllm.utils.lr_schedules import get_power_schedule

        return get_power_schedule(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            power_start_step=_steps(
                sched_cfg.get("power_start_step", num_warmup_steps)
            ),
            exponent=float(sched_cfg.get("exponent", 0.51)),
            num_decay_steps=num_decay_steps,
            min_lr_ratio=min_lr_ratio,
        )
    if scheduler_name == "cosine":
        from transformers.optimization import get_cosine_schedule_with_warmup

        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    elif scheduler_name == "linear":
        from transformers.optimization import get_linear_schedule_with_warmup

        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    elif scheduler_name == "constant":
        from transformers.optimization import get_constant_schedule

        return get_constant_schedule(optimizer)
    elif scheduler_name == "constant_warmup":
        from transformers.optimization import get_constant_schedule_with_warmup

        return get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps
        )
    elif scheduler_name == "inverse_sqrt":
        from transformers.optimization import get_inverse_sqrt_schedule

        return get_inverse_sqrt_schedule(
            optimizer, num_warmup_steps=num_warmup_steps
        )
    else:
        raise ValueError(
            f"Unknown scheduler: {scheduler_name}. "
            f"Supported: cosine, linear, constant, constant_warmup, "
            f"inverse_sqrt, wsd, power"
        )
