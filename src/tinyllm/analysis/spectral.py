import time

import torch


def compute_svd_and_variance(W: torch.Tensor, thresholds=(0.95, 0.99)):
    """
    Compute SVD and variance explained for a weight matrix.

    The **condition number** (σ_max / σ_min) measures how ill-conditioned
    the weight matrix is. Values >> 1 indicate near-singular directions
    that can cause training instability. Values ≈ 1 indicate isotropic,
    well-conditioned transformations. Infinity means the matrix is
    rank-deficient (at least one zero singular value).

    Args:
        W: 2D weight tensor (M, N).
        thresholds: Tuple of variance thresholds to compute.

    Returns:
        Dict with singular values, eigenvalues, variance ratios,
        number of eigenvalues needed for each threshold, and condition number.
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

    # Condition number = σ_max / σ_min.
    #   ~1  → isotropic, well-conditioned (all singular values similar)
    #   >>1 → ill-conditioned, near-singular (some directions strongly suppressed)
    #   inf → rank-deficient (exact zero singular value)
    result["condition_number"] = (
        (S[0] / S[-1]).item() if S[-1] > 0 else float("inf")
    )

    return result


def compute_weightwatcher_alpha(W: torch.Tensor, xmin=None):
    """
    Compute WeightWatcher power-law alpha exponent for a weight matrix.

    The alpha parameter measures how heavy-tailed the eigenvalue distribution is.
    It is estimated by fitting P(λ) ~ λ^(−α) to the tail of the empirical
    spectrum using the Hill estimator with KS-based cutoff selection.

    Interpretation:
        α < 2   — Heavy-tailed; layer has learned structured features.
                  Very low α (< 1.5) may indicate memorization/overfitting.
        α 2–4   — Well-trained with good generalization (optimal range).
        α 4–6   — Moderate structure, somewhat undertrained.
        α 6–10  — Near random initialization; limited learning.
        α → ∞   — White noise (all eigenvalues equal); untrained.

    Args:
        W: 2D weight tensor (M, N).
        xmin: Optional lower bound for the power-law tail fit.
              If None, the Hill estimator searches for the optimal cutoff.

    Returns:
        Alpha exponent (float). Higher = more random, lower = more structured.
    """
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
