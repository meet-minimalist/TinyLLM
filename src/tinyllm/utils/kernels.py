"""
Optional integration with optimized Triton kernels.

Currently supported:
  - liger-kernel: fused RMSNorm, RoPE, SwiGLU, CrossEntropy, LayerNorm
    (https://github.com/linkedin/Liger-Kernel)
  - flash-attn:   faster attention via flash_attn package
    (https://github.com/Dao-AILab/flash-attention)

All imports are lazy — these packages are NOT required.
"""

import os

import torch
import torch.nn as nn

# Liger requires Triton, which has no official Windows build.
# triton-windows exists but doesn't support all Liger ops (BF16 fused CE fails).
_LIGER_SUPPORTED = os.name != "nt"


def apply_kernel_patches(model: nn.Module, config: dict) -> nn.Module:
    """
    Apply all enabled kernel patches to a model.

    Called after model construction in train.py.
    Config keys under 'kernels':
        use_liger: bool   — patch with liger-kernel fused ops (Linux/WSL2 only)
        use_flash_attn: bool  — uses F.scaled_dot_product_attention
            (handled per-layer via the `flash` parameter, not here)

    Returns the patched model (or original if nothing to apply).
    """
    cfg = (
        config.get("kernels", {})
        if isinstance(config, dict)
        else getattr(config, "kernels", {})
    )
    if not cfg:
        return model

    if cfg.get("use_liger", False):
        if _LIGER_SUPPORTED:
            model = _patch_liger(model)
        else:
            from src.tinyllm.logger.logger_utils import logger

            logger.warning(
                "Liger kernels skipped on Windows (triton-windows does not support all ops)."
            )

    if cfg.get("use_flash_attn", False):
        _patch_attention_flash(model)

    return model


def _patch_liger(model: nn.Module) -> nn.Module:
    """
    Patch model with liger-kernel fused Triton implementations.

    Swaps:
      RMSNorm  -> LigerRMSNorm  (fused RMS + scale kernel)
      GatedFFN -> LigerSwiGLUMLP  (fused SiLU(gate)*up kernel, avoids separate activation pass)

    `apply_liger_kernel_to_model` targets HuggingFace class names and won't recognize
    our custom classes, so we iterate and replace explicitly.
    """
    try:
        import types
        from liger_kernel.transformers import LigerRMSNorm
        from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
    except ImportError:
        return model

    for name, module in list(model.named_modules()):
        # Navigate to parent module
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        child_name = parts[-1]

        if module.__class__.__name__ == "RMSNorm":
            liger_norm = LigerRMSNorm(module.weight.shape[0], eps=module.eps)
            liger_norm.weight = module.weight  # share the parameter tensor
            setattr(parent, child_name, liger_norm)

        elif module.__class__.__name__ == "GatedFFN":
            # LigerSwiGLUMLP expects a config-like object with hidden_size,
            # intermediate_size, and hidden_act attributes.
            hidden = module.gate.in_features
            intermediate = module.gate.out_features
            cfg = types.SimpleNamespace(
                hidden_size=hidden,
                intermediate_size=intermediate,
                hidden_act="silu",
            )
            liger_ffn = LigerSwiGLUMLP(cfg)
            liger_ffn.gate_proj.weight = module.gate.weight
            liger_ffn.up_proj.weight = module.up.weight
            liger_ffn.down_proj.weight = module.down.weight
            setattr(parent, child_name, liger_ffn)

    return model


def _patch_attention_flash(model: nn.Module):
    """Sets flash=True on all attention modules in the model."""
    # Check for flash_attn_varlen_func specifically — not just the package.
    # FA4 installs as flash_attn but without this function; flex_attention
    # is the fallback in that case (checked at forward time in gqa/mha).
    has_varlen = False
    try:
        from flash_attn import flash_attn_varlen_func  # noqa: F401

        has_varlen = True
    except (ImportError, AttributeError):
        pass

    has_flex = False
    try:
        from torch.nn.attention.flex_attention import (
            flex_attention,
        )  # noqa: F401

        has_flex = True
    except ImportError:
        pass

    if not has_varlen and not has_flex:
        print(
            "Neither flash_attn nor flex_attention available; using SDPA fallback."
        )

    for module in model.modules():
        if hasattr(module, "flash"):
            module.flash = True


def try_liger_fused_ce():
    """Return LigerFusedLinearCrossEntropyLoss class if available (Linux/Triton only).

    Fuses lm_head matmul + softmax + cross-entropy in one Triton kernel,
    never materializing the [S, vocab_size] logits tensor. Saves ~50% peak
    memory on the loss step for large vocabularies.

    Usage: loss = fused_ce(hidden, lm_head_weight, labels)
      - hidden:          [B*S, d_model]  (output of final norm)
      - lm_head_weight:  [vocab_size, d_model]
      - labels:          [B*S]  (flat target token ids)
    """
    if not _LIGER_SUPPORTED:
        return None
    try:
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

        return LigerFusedLinearCrossEntropyLoss
    except ImportError:
        return None


class _CCEFusedCELoss:
    """
    Wraps cut-cross-entropy's linear_cross_entropy to match Liger's call signature.

    Fuses lm_head matmul + softmax + CE without materializing [S, vocab] logits.
    Works on Windows (no Triton required).
    """

    def __init__(self, ignore_index=-100, label_smoothing=0.0):
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def __call__(self, hidden, weight, labels):
        from cut_cross_entropy import linear_cross_entropy

        return linear_cross_entropy(
            hidden, weight, labels, ignore_index=self.ignore_index
        )


def try_fused_ce(ignore_index=-100, label_smoothing=0.0, use_liger=False):
    """
    Return an instantiated fused linear+CE loss, trying Liger then cut-cross-entropy.

    Both avoid materializing the [B*S, vocab_size] logits tensor.
    Call signature: loss = fused_ce(hidden [B*S, D], lm_head_weight [vocab, D], labels [B*S])

    use_liger: whether to attempt the Liger backend (Linux/Triton only).
    Returns (instance, backend_name) or (None, None) if neither is available.
    """
    if use_liger and _LIGER_SUPPORTED:
        cls = try_liger_fused_ce()
        if cls is not None:
            return (
                cls(ignore_index=ignore_index, label_smoothing=label_smoothing),
                "liger",
            )

    # CCE does not support label_smoothing — fall back to standard CE so smoothing is honoured.
    if label_smoothing > 0:
        return None, None

    try:
        from cut_cross_entropy import linear_cross_entropy  # noqa: F401

        return (
            _CCEFusedCELoss(ignore_index=ignore_index, label_smoothing=0.0),
            "cce",
        )
    except ImportError:
        return None, None


def audit_kernel_patches(
    model: nn.Module, fused_ce=None, train_config=None
) -> None:
    """
    Log the full optimisation status of the model — both architecture choices
    and optional kernel patches. Inspects actual module state, not config intent.
    """
    from src.tinyllm.logger.logger_utils import logger

    tally = {}
    for _, module in model.named_modules():
        cls = module.__class__.__name__
        tally[cls] = tally.get(cls, 0) + 1

    # Liger patch status
    rms_orig = tally.get("RMSNorm", 0)
    rms_liger = tally.get("LigerRMSNorm", 0)
    ffn_orig = tally.get("GatedFFN", 0)
    ffn_liger = tally.get("LigerSwiGLUMLP", 0)

    def _liger(liger, orig):
        if liger > 0 and orig == 0:
            return f"ACTIVE ({liger} swapped)"
        if liger == 0 and orig > 0:
            return f"inactive ({orig} original)"
        if liger > 0 and orig > 0:
            return f"PARTIAL ({liger} swapped, {orig} remain)"
        return "n/a"

    # Architecture choices (from module class names)
    has_gqa = tally.get("GroupedQueryAttention", 0) > 0
    has_gated_ffn = (
        tally.get("GatedFFN", 0) + tally.get("LigerSwiGLUMLP", 0)
    ) > 0
    has_rms = (tally.get("RMSNorm", 0) + tally.get("LigerRMSNorm", 0)) > 0
    flash_active = any(
        getattr(m, "flash", False)
        for m in model.modules()
        if hasattr(m, "flash")
    )
    is_compiled = hasattr(model, "_orig_mod")

    # Precision from config
    precision = "unknown"
    if train_config is not None:
        cfg = (
            train_config
            if isinstance(train_config, dict)
            else vars(train_config) if hasattr(train_config, "__dict__") else {}
        )
        precision = cfg.get(
            "precision", getattr(train_config, "precision", "unknown")
        )

    logger.info("=== Training Optimisation Status ===")
    logger.info("  -- Architecture --")
    logger.info(f"  Precision       : {precision}")
    logger.info(f"  GQA             : {'ACTIVE' if has_gqa else 'inactive'}")
    logger.info(
        f"  SwiGLU (GatedFFN): {'ACTIVE' if has_gated_ffn else 'inactive'}"
    )
    logger.info(f"  RMSNorm         : {'ACTIVE' if has_rms else 'inactive'}")
    logger.info(
        f"  Flash Attention : {'ACTIVE' if flash_active else 'inactive'}"
    )
    logger.info(
        f"  torch.compile   : {'ACTIVE' if is_compiled else 'inactive'}"
    )
    logger.info("  -- Liger fused kernels (Linux/Triton only) --")
    logger.info(f"  RMSNorm fused   : {_liger(rms_liger, rms_orig)}")
    logger.info(f"  SwiGLU fused    : {_liger(ffn_liger, ffn_orig)}")
    logger.info(
        f"  Fused CE loss   : {'ACTIVE' if fused_ce is not None else 'inactive'}"
    )
    logger.info("====================================")


def check_kernel_status() -> dict:
    """Report which optional kernel packages are available."""
    status = {
        "flash_attn_in_torch": hasattr(
            torch.nn.functional, "scaled_dot_product_attention"
        ),
        "flash_attn_package": False,
        "liger_kernel": False,
    }
    try:
        import flash_attn

        status["flash_attn_package"] = True
    except ImportError:
        pass
    try:
        import liger_kernel

        status["liger_kernel"] = len(dir(liger_kernel)) > 0
    except ImportError:
        pass
    return status
