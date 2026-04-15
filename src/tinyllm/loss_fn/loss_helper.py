"""
# @ Author: Meet Patel
# @ Create Time: 2025-03-30 15:29:12
# @ Modified by: Meet Patel
# @ Modified time: 2026-01-12 20:42:43
# @ Description:
"""

import torch
import torch.nn as nn


def compute_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_smoothing: float = 0.0,
    ignore_index: int = -100,
):
    """
    Computes cross entropy loss based on provided logits and labels.
    Note: We will not be computing loss for all tokens of all batches.
            Instead, we will only consider non-padded tokens for this.

    Args:
        logits (torch.Tensor): Logits from model without Softmax. Shape: [batch, seq, vocab_size].
        labels (torch.Tensor): Labels which are one token shifted input_ids. Shape: [batch, seq]
        label_smoothing (float): Intensity of label smoothing to apply. Defaults to 0.0.
        ignore_index (int): Index of labels to ignore while performing softmax. Defaults to -100.

    Returns:
        torch.Tensor: Cross Entropy Loss value for batch of logits and labels.
    """
    batch_size = logits.shape[0]
    logits = logits.view(-1, logits.shape[2])
    labels = labels.view(-1)

    # We would take mean across all sequence length and all batches.
    loss = nn.functional.cross_entropy(
        logits,
        labels,
        reduction="mean",
        label_smoothing=label_smoothing,
        ignore_index=ignore_index,
    )

    return loss
