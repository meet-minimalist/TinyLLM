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
