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
            w = p.detach().float().flatten()
            w_abs = w.abs()
            stats[name] = {
                "_hist": np.histogram(w.cpu().numpy(), bins=num_bins),
                "snr": (w_abs.mean() / (w_abs.std() + 1e-8)).item(),
            }
    stats["_time"] = time.perf_counter() - t0
    return stats
