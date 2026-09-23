"""Exclusive Self-Attention (XSA, Zhai, arXiv:2603.09078).

A two-line post-processing step applied to standard (causal) self-attention's
output. For each token, XSA removes the projection of the attention output
onto that token's own value vector::

    z_i = y_i - (y_i . v_i) * v_i / ||v_i||^2

so the corrected output z_i no longer contains v_i itself, nor any component
of the context correlated with it — attention is forced to explain the token
using *other* positions instead of restating its own value. The correction
runs per head, after the softmax-weighted sum and before merging heads back
into the model dimension.
"""

import torch
import torch.nn.functional as F


def apply_xsa(y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Remove each token's self-value component from its attention output.

    Args:
        y: attention output, ``[..., seq_len, head_dim]``.
        v: the *same* token's value vector, one per query head — under GQA
            this is the shared kv-head value repeated across its query-head
            group, so ``v`` must already be expanded to ``y``'s head count.

    Both tensors keep their input dtype; the normalize is done in the input
    dtype like the rest of attention (unlike nGPT's weight-sphere projection,
    this has no compounding-error concern — it runs once per layer, not
    every step, and its result is immediately consumed by out_proj).
    """
    v_n = F.normalize(v, p=2, dim=-1)
    return y - (y * v_n).sum(dim=-1, keepdim=True) * v_n
