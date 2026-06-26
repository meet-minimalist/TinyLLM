import time

import numpy as np
import torch


def compute_weight_stats(model, num_bins: int):
    """Compute per-parameter weight histograms for WandB logging.

    Returns:
        Dict mapping param name -> {"_hist": np.histogram result}
        plus "_time" key.
    """
    t0 = time.perf_counter()
    stats = {}
    for name, p in model.named_parameters():
        if p.ndim < 1:
            continue
        with torch.no_grad():
            stats[name] = {
                "_hist": np.histogram(
                    p.detach().float().flatten().cpu().numpy(), bins=num_bins
                )
            }
    stats["_time"] = time.perf_counter() - t0
    return stats


def compute_gradient_norms(model):
    """Per-parameter and global gradient norms."""
    t0 = time.perf_counter()
    total_norm = 0.0
    layer_norms = {}
    n_zero = 0
    n_total = 0
    n_exploded = 0

    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        param_norm = p.grad.norm().item()
        layer_norms[name] = param_norm
        total_norm += param_norm**2
        n_total += 1
        if param_norm == 0:
            n_zero += 1
        if param_norm > 1e4:
            n_exploded += 1

    return {
        "global/gradient_norm": total_norm**0.5,
        "global/exploded_gradients": n_exploded,
        "global/zero_gradients": n_zero,
        "global/total_params_with_grad": n_total,
        "_layer_norms": layer_norms,
        "_time": time.perf_counter() - t0,
    }
