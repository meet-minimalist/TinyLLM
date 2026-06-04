import time

import torch


def compute_svd_and_variance(W: torch.Tensor, thresholds=(0.95, 0.99)):
    """
    Compute SVD and variance explained for a weight matrix.

    Args:
        W: 2D weight tensor (M, N).
        thresholds: Tuple of variance thresholds to compute.

    Returns:
        Dict with singular values, eigenvalues, variance ratios,
        and number of eigenvalues needed for each threshold.
    """
    W = W.float()
    t0 = time.perf_counter()
    S = torch.linalg.svdvals(W)
    t_svd = time.perf_counter() - t0

    evals = S**2
    total = evals.sum()
    cum_ratio = evals.cumsum(0) / total

    result = {
        "singular_values": S.detach().cpu(),
        "eigenvalues": evals.detach().cpu(),
        "cumulative_variance_ratio": cum_ratio.detach().cpu(),
        "total_variance": total.item(),
        "num_singular_values": S.shape[0],
        "time_svd": t_svd,
    }

    for t in thresholds:
        n = int((cum_ratio < t).sum().item()) + 1
        result[f"n_for_{int(t*100)}pct"] = n
        result[f"ratio_for_{int(t*100)}pct"] = min(1.0, n / max(1, S.shape[0]))

    result["condition_number"] = (
        (S[0] / S[-1]).item() if S[-1] > 0 else float("inf")
    )

    return result


def compute_weightwatcher_alpha(W: torch.Tensor, xmin=None):
    """Compute WeightWatcher power-law alpha exponent."""
    W = W.float()
    sv = torch.linalg.svdvals(W)
    evals = sv * sv
    return _fit_powerlaw_alpha(evals, xmin)


def _fit_powerlaw_alpha(data: torch.Tensor, xmin=None):
    """Fit power law alpha using Hill estimator + KS minimization."""
    device = data.device
    data = data.double()
    data, _ = torch.sort(data)

    if xmin is not None:
        i_min = torch.argmin(torch.abs(data - xmin)).item()
    else:
        i_min = 0

    data = data[i_min:]
    N = data.shape[0]
    if N < 2:
        return 0.0

    log_data = torch.log(data)
    suffix_sum = torch.flip(
        torch.cumsum(torch.flip(log_data, dims=[0]), dim=0), dims=[0]
    )

    i_idx = torch.arange(N - 1, device=device, dtype=torch.float64)
    n = N - i_idx
    alphas = 1.0 + n / (suffix_sum[: N - 1] - n * log_data[: N - 1])

    Ds = torch.ones(N - 1, device=device, dtype=torch.float64)
    valid = alphas > 1.0
    for k in valid.nonzero(as_tuple=True)[0].tolist():
        tail = data[k:]
        n_k = tail.shape[0]
        j_local = torch.arange(n_k, device=device, dtype=torch.float64)
        cdf_model = 1.0 - (tail / data[k]) ** (-alphas[k] + 1.0)
        cdf_empirical = j_local / n_k
        Ds[k] = torch.max(torch.abs(cdf_model - cdf_empirical))

    best = torch.argmin(Ds).item()
    return alphas[best].item()
