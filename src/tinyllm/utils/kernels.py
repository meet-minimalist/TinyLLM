"""
Optional integration with optimized Triton kernels.

Currently supported:
  - liger-kernel: fused RMSNorm, RoPE, SwiGLU, CrossEntropy, LayerNorm
    (https://github.com/linkedin/Liger-Kernel)
  - flash-attn:   faster attention via flash_attn package
    (https://github.com/Dao-AILab/flash-attention)

All imports are lazy — these packages are NOT required.
"""

import torch
import torch.nn as nn


def apply_kernel_patches(model: nn.Module, config: dict) -> nn.Module:
    """
    Apply all enabled kernel patches to a model.

    Called after model construction in train.py.
    Config keys under 'kernels':
        use_liger: bool   — patch with liger-kernel fused ops
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
        model = _patch_liger(model)

    if cfg.get("use_flash_attn", False):
        _patch_attention_flash(model)

    return model


def _patch_liger(model: nn.Module) -> nn.Module:
    """
    Patch model with liger-kernel fused implementations.

    Replaces RMSNorm, LayerNorm, RoPE, SwiGLU, etc. with fused Triton kernels.
    liger-kernel's apply_liger_kernel_to_model handles all supported architectures.
    """
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_model

        model = apply_liger_kernel_to_model(model)
        return model
    except ImportError:
        return model
    except Exception:
        return model


def _patch_attention_flash(model: nn.Module):
    """Sets flash=True on all attention modules in the model."""
    has_pkg = False
    try:
        import flash_attn  # noqa: F401

        has_pkg = True
    except ImportError:
        print("flash_attn package not found. Flash attention will not be used.")
        pass
    if not has_pkg:
        return
    for module in model.modules():
        if hasattr(module, "flash"):
            module.flash = True


def try_liger_cross_entropy():
    """
    Return liger-kernel's fused CrossEntropyLoss if available.

    FusedLinearCrossEntropy avoids materializing the full logits tensor
    by chunking the computation. Can reduce memory by ~60% for large vocab.
    """
    try:
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

        return LigerFusedLinearCrossEntropyLoss
    except ImportError:
        return None


def try_liger_rmsnorm():
    """Return liger-kernel's fused RMSNorm class if available."""
    try:
        from liger_kernel.transformers import LigerRMSNorm

        return LigerRMSNorm
    except ImportError:
        return None


def try_liger_swiglu():
    """Return liger-kernel's fused SwiGLU MLP class if available."""
    try:
        from liger_kernel.transformers.swiglu import LigerSwiGLUMLP

        return LigerSwiGLUMLP
    except ImportError:
        return None


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
