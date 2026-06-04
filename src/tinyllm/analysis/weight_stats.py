import time

import torch


def compute_weight_stats(model):
    """
    Compute basic statistics for every weight tensor in the model.

    Returns:
        Dict mapping weight name -> {mean, std, min, max, norm, sparsity, numel}
    """
    t0 = time.perf_counter()
    stats = {}

    for name, p in model.named_parameters():
        if p.ndim < 1:
            continue
        with torch.no_grad():
            stats[name] = {
                "mean": p.mean().item(),
                "std": p.std().item(),
                "min": p.min().item(),
                "max": p.max().item(),
                "norm": p.norm().item(),
                "numel": p.numel(),
                "shape": list(p.shape),
            }
            if p.ndim >= 2:
                stats[name]["sparsity"] = (p == 0).float().mean().item()

    stats["_time"] = time.perf_counter() - t0
    return stats


def compute_gradient_norms(model):
    """
    Compute per-parameter and global gradient norms to detect explosion/vanishing.

    Returns:
        Dict with global stats and per-layer gradient norms.
    """
    t0 = time.perf_counter()
    total_norm = 0.0
    layer_norms = {}
    n_zero = 0
    n_total = 0
    n_exploded = 0

    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        grad = p.grad
        param_norm = grad.norm().item()
        layer_norms[name] = param_norm
        total_norm += param_norm**2
        n_total += 1
        if param_norm == 0:
            n_zero += 1
        if param_norm > 1e4:
            n_exploded += 1

    total_norm = total_norm**0.5
    elapsed = time.perf_counter() - t0

    return {
        "global/gradient_norm": total_norm,
        "global/exploded_gradients": n_exploded,
        "global/zero_gradients": n_zero,
        "global/total_params_with_grad": n_total,
        "_layer_norms": layer_norms,
        "_time": elapsed,
    }
