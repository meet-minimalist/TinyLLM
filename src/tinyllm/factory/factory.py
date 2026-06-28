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
):
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
            f"Supported: cosine, linear, constant, constant_warmup, inverse_sqrt"
        )
