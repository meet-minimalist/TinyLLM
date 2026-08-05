"""
fp8 training support (H100 / Hopper) via torchao float8.

fp8 is an *add-on* to the bf16 path, not a replacement for it. We keep the
model's master weights and the optimizer state in fp32 and keep bf16 autocast
for all non-matmul ops (norms, softmax, residuals). ``torchao.float8`` swaps
eligible ``nn.Linear`` modules for ``Float8Linear``, which casts only the
*matmul inputs* to fp8 on each step (dynamic scaling).

Sensitive layers are deliberately left in bf16:
  * ``lm_head`` (and anything matched by ``filter_fqns``) — the output
    projection to vocab is the most numerically sensitive matmul and aligns
    with the fused cross-entropy path, which also keeps ``lm_head`` in bf16.
  * any Linear whose in/out features are not divisible by 16 — fp8 tensor
    cores require 16-aligned dims.

The conversion must run *before* ``torch.compile`` so the compiler traces the
already-swapped ``Float8Linear`` modules.
"""

from typing import Tuple

import torch
import torch.nn as nn

from src.tinyllm.logger.logger_utils import logger

# Minimum CUDA compute capability for fp8 tensor cores (Ada SM89 / Hopper SM90).
_MIN_FP8_CAPABILITY = (8, 9)

_SUPPORTED_RECIPES = ("tensorwise", "rowwise")


def maybe_convert_to_fp8(
    model: nn.Module, train_config
) -> Tuple[nn.Module, bool]:
    """Optionally convert ``model``'s Linears to fp8 training modules.

    Reads the ``fp8`` block from ``train_config``:

        fp8:
          enabled: false           # master switch
          recipe: "tensorwise"     # tensorwise (default) | rowwise
          filter_fqns: ["lm_head"] # module-name substrings kept in bf16

    Returns ``(model, fp8_enabled)``. Every failure mode (disabled, no CUDA,
    pre-Hopper GPU, torchao missing) is a graceful no-op that logs and returns
    the model unchanged — fp8 is opt-in and must never break a run.
    """
    fp8_cfg = _get(train_config, "fp8", {}) or {}
    if not _get(fp8_cfg, "enabled", False):
        return model, False

    # --- Hardware / dependency guards (graceful no-op) ---
    if not torch.cuda.is_available():
        logger.warning(
            "fp8 requested but CUDA is unavailable; running without fp8."
        )
        return model, False

    capability = torch.cuda.get_device_capability()
    if capability < _MIN_FP8_CAPABILITY:
        logger.warning(
            f"fp8 requested but GPU compute capability {capability} < "
            f"{_MIN_FP8_CAPABILITY} (needs Ada SM89+ / Hopper H100); "
            "running without fp8."
        )
        return model, False

    try:
        from torchao.float8 import (
            convert_to_float8_training,
            Float8LinearConfig,
        )
    except ImportError:
        logger.warning(
            "fp8 requested but torchao is not installed. Falling back to "
            "bf16. Install with: pip install torchao"
        )
        return model, False

    # --- Recipe selection ---
    recipe = _get(fp8_cfg, "recipe", "tensorwise")
    if recipe not in _SUPPORTED_RECIPES:
        raise ValueError(
            f"Unknown fp8 recipe '{recipe}'. Supported: {_SUPPORTED_RECIPES}"
        )
    fp8_config = Float8LinearConfig.from_recipe_name(recipe)

    # --- Which Linears to convert ---
    filter_fqns = list(_get(fp8_cfg, "filter_fqns", ["lm_head"]))

    def module_filter_fn(module: nn.Module, fqn: str) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        # fp8 tensor cores need 16-aligned dims.
        if module.in_features % 16 != 0 or module.out_features % 16 != 0:
            return False
        # Keep sensitive / explicitly-excluded layers in bf16.
        if any(skip in fqn for skip in filter_fqns):
            return False
        return True

    convert_to_float8_training(
        model, config=fp8_config, module_filter_fn=module_filter_fn
    )

    num_converted = sum(
        1 for m in model.modules() if type(m).__name__ == "Float8Linear"
    )
    logger.info(
        f"fp8 enabled ({recipe}): {num_converted} Linear layers converted "
        f"(GPU capability {capability}, kept-in-bf16 fqns={filter_fqns})."
    )
    return model, True


def _get(cfg, key, default):
    """Read ``key`` from a Box/dict/attr-style config, else ``default``."""
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        try:
            val = cfg.get(key, default)
            return val if val is not None else default
        except TypeError:
            pass
    return getattr(cfg, key, default)
