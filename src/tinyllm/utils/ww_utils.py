# Code to calculate alpha based on weight watcher's method.
# Reference: https://github.com/CalculatedContent/WeightWatcher

"""
Minimal script to compute alpha for FC weight matrix — PyTorch GPU version.
"""

import torch
import numpy as np


def compute_alpha(W: torch.Tensor) -> float:
    """
    Compute the power law alpha exponent for a fully connected (FC) weight matrix.

    Parameters
    ----------
    W : torch.Tensor
        2D weight matrix of shape (N, M), on CPU or GPU.

    Returns
    -------
    alpha : float
    """
    # Ensure float32 for SVD stability
    W = W.float()

    # torch.linalg.svdvals returns singular values in descending order
    sv = torch.linalg.svdvals(W)  # shape: (min(N,M),)
    M = min(W.shape)
    sv = sv[:M]  # already sorted descending; keep top-M

    evals = sv * sv
    return fit_powerlaw_alpha(evals)


def fit_powerlaw_alpha(data: torch.Tensor, xmin: float | None = None) -> float:
    """
    Fit power law to eigenvalues using the Hill / KS-statistic method.

    Parameters
    ----------
    data  : torch.Tensor  –  eigenvalue array (unsorted is fine)
    xmin  : float, optional  –  minimum value to start fit

    Returns
    -------
    alpha : float
    """
    device = data.device

    # Sort ascending on GPU, then work entirely on GPU until the final scalar
    data = data.double()
    data, _ = torch.sort(data)

    if xmin is not None:
        i_min = torch.argmin(torch.abs(data - xmin)).item()
    else:
        i_min = 0

    data = data[i_min:]  # slice from i_min to end (inclusive)
    N = data.shape[0]

    log_data = torch.log(data)  # shape: (N,)

    # We'll compute alpha_i and D_i for i in 0 … N-2
    num_candidates = N - 1

    # Suffix sums of log_data: suffix_sum[i] = sum(log_data[i:])
    # Computed via cumsum on the reversed tensor
    suffix_sum = torch.flip(
        torch.cumsum(torch.flip(log_data, dims=[0]), dim=0), dims=[0]
    )  # shape: (N,)

    # n_i = N - i  (number of points from i onward)
    i_idx = torch.arange(num_candidates, device=device, dtype=torch.float64)
    n = N - i_idx  # shape: (num_candidates,)

    # Hill estimator: alpha = 1 + n / (sum_log[i:] - n * log_data[i])
    alphas = 1.0 + n / (
        suffix_sum[:num_candidates] - n * log_data[:num_candidates]
    )

    # KS statistic — vectorised over all candidates
    # For candidate i with exponent alpha_i, xmin_i = data[i]:
    #   D_i = max_j>=i |  1 - (data[j]/xmin_i)^(-alpha_i+1)  -  j_local/n_i  |
    # We only compute D where alpha > 1; others stay at 1.0 (large sentinel)

    Ds = torch.ones(num_candidates, device=device, dtype=torch.float64)

    valid_mask = alphas > 1.0
    valid_idx = valid_mask.nonzero(as_tuple=True)[0]

    for k in valid_idx.tolist():
        xmin_k = data[k]
        alpha_k = alphas[k]
        tail = data[k:]  # shape: (n_k,)
        n_k = tail.shape[0]
        j_local = torch.arange(n_k, device=device, dtype=torch.float64)

        cdf_model = 1.0 - (tail / xmin_k) ** (-alpha_k + 1.0)
        cdf_empirical = j_local / n_k
        Ds[k] = torch.max(torch.abs(cdf_model - cdf_empirical))

    best_i = torch.argmin(Ds).item()
    return alphas[best_i].item()


# ── entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    W = torch.randn(1000, 1000, device=device)  # dummy data for testing
    alpha = compute_alpha(W)
    print(f"Alpha: {alpha:.4f}")
