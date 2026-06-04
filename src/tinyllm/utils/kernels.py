"""
Optional integration with optimized Triton kernels.

Supports:
  - liger-kernel: fused CrossEntropy, RoPE, SwiGLU, RMSNorm, LayerNorm
  - flash-attn:   faster attention via flash_attn package (not just PyTorch's SDPA)

Enable via config:
  kernels:
    use_flash_attn: true       # uses F.scaled_dot_product_attention
    use_liger: true            # uses liger-kernel fused ops

All imports are lazy — these packages are NOT required to run the codebase.
"""

import torch


def try_liger_patch(model):
    """
    Attempt to patch model modules with liger-kernel fused implementations.

    Returns the patched model (or original if liger not available).
    """
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_model

        apply_liger_kernel_to_model(model)
        return model
    except ImportError:
        return model
    except Exception:
        return model


def try_liger_cross_entropy():
    """
    Return liger-kernel's fused CrossEntropyLoss if available,
    otherwise return None (fallback to standard CE).
    """
    try:
        from liger_kernel.transformers import LigerCrossEntropyLoss

        return LigerCrossEntropyLoss
    except ImportError:
        return None


def try_flash_attn(attn_fn):
    """
    Attempt to replace attention with flash-attn package.

    This is more flexible than PyTorch's SDPA (supports GQA natively
    without repeating kv heads, sliding window, etc.).
    """
    try:
        from flash_attn.flash_attention import FlashMHA

        return FlashMHA
    except ImportError:
        return None


def check_kernel_status():
    """Return dict indicating which optional kernels are available."""
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

        status["liger_kernel"] = True
    except ImportError:
        pass
    return status
