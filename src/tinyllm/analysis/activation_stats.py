import time
from typing import Optional

import torch


class ActivationCapture:
    """
    Forward hook that captures intermediate activations from attention layers.

    Attach to any attention layer's forward method. Works with any module
    that returns (output, metadata_dict) where metadata_dict contains
    keys like "q", "k", "v", "attn_weights", "o_proj_out".
    """

    def __init__(self, name: str, capture_freq: int = 1):
        self.name = name
        self.capture_freq = capture_freq
        self.call_count = 0
        self.captured = {}

    def __call__(self, module, input_args, output):
        self.call_count += 1
        if self.call_count % self.capture_freq != 0:
            return

        _, metadata = output
        self.captured = {
            k: v.detach()
            for k, v in metadata.items()
            if isinstance(v, torch.Tensor)
        }


def compute_activation_stats(captures: dict) -> dict:
    """
    Compute statistics from captured activation tensors.

    Args:
        captures: Dict mapping layer_name -> {tensor_name -> tensor}

    Returns:
        Flattened dict suitable for WandB logging.
    """
    t0 = time.perf_counter()
    stats = {}

    for layer_name, tensors in captures.items():
        for tensor_name, tensor in tensors.items():
            with torch.no_grad():
                key = f"{layer_name}/{tensor_name}"
                stats[f"{key}/mean"] = tensor.mean().item()
                stats[f"{key}/std"] = tensor.std().item()
                stats[f"{key}/min"] = tensor.min().item()
                stats[f"{key}/max"] = tensor.max().item()
                stats[f"{key}/norm"] = tensor.norm().item()
                stats[f"{key}/shape"] = list(tensor.shape)

    stats["_time"] = time.perf_counter() - t0
    return stats
