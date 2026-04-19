"""
Factory functions for creating models and LR schedulers from config.
"""

from box import Box
from torch.optim.lr_scheduler import (
    LambdaLR,
    CosineAnnealingLR,
    LinearLR,
    ConstantLR,
)

from src.tinyllm.factory.registry import MODEL_REGISTRY
from src.tinyllm.logger.logger_utils import logger

# Built-in schedulers (no external dependency)
_builtin_schedulers = {
    "cosine": CosineAnnealingLR,
    "linear": LinearLR,
    "constant": ConstantLR,
}


def model_factory(model_config: Box):
    """
    Create a model instance based on the model_type in config.

    Args:
        model_config: Model architecture config (Box).

    Returns:
        nn.Module: Model instance.
    """
    if not hasattr(model_config, "model_type"):
        raise ValueError("model_type must be specified in model_config.")

    # Ensure model classes are registered by importing the models module
    import src.tinyllm.models  # noqa: F401 — triggers registry

    return MODEL_REGISTRY.get(model_config.model_type)(model_config)


def lr_scheduler_factory(
    scheduler_name: str,
    optimizer,
    num_training_steps: int,
    num_warmup_steps: int = 0,
) -> LambdaLR:
    """
    Create a learning rate scheduler.

    Args:
        scheduler_name: Name of scheduler type.
        optimizer: PyTorch optimizer.
        num_training_steps: Total training steps.
        num_warmup_steps: Number of warmup steps.

    Returns:
        LR scheduler instance.
    """
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
            f"Supported: {list(_builtin_schedulers.keys())}"
        )
