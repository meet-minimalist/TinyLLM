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
