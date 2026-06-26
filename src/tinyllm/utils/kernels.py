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
    Patch model with liger-kernel fused Triton implementations.

    Swaps:
      RMSNorm  → LigerRMSNorm  (fused RMS + scale kernel)
      GatedFFN → LigerSwiGLUMLP  (fused SiLU(gate)*up kernel, avoids separate activation pass)

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


def try_liger_fused_ce():
    """
    Return liger-kernel's LigerFusedLinearCrossEntropyLoss if available.

    Fuses lm_head matmul + softmax + cross-entropy in one Triton kernel,
    never materializing the [S, vocab_size] logits tensor. Saves ~50% peak
    memory on the loss step for large vocabularies.

    Usage: loss = fused_ce(hidden, lm_head_weight, labels)
      - hidden:          [B*S, d_model]  (output of final norm)
      - lm_head_weight:  [vocab_size, d_model]
      - labels:          [B*S]  (flat target token ids)
    """
    try:
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

        return LigerFusedLinearCrossEntropyLoss
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
