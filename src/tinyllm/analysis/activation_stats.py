import time

import torch


class ForwardHookCapture:
    """General-purpose forward hook store.

    Call make_hook(name) to get a hook closure and register it via
    module.register_forward_hook().  All captured outputs land in
    self.captured keyed by name.

    Two modes selected per-hook:
      unpack_metadata=False (default): stores the plain tensor output.
      unpack_metadata=True: expects (tensor, dict) and stores the dict —
        used for attention layers that return metadata alongside the output.
    """

    def __init__(self, capture_freq: int = 1):
        self.capture_freq = capture_freq
        self._call_counts = {}
        self.captured = {}  # name -> tensor  OR  name -> {str: tensor}

    def make_hook(self, name: str, unpack_metadata: bool = False):
        self._call_counts[name] = 0

        def hook(module, input, output):
            self._call_counts[name] = self._call_counts.get(name, 0) + 1
            if self._call_counts[name] % self.capture_freq != 0:
                return
            if unpack_metadata:
                if isinstance(output, (tuple, list)) and len(output) == 2:
                    _, meta = output
                    if isinstance(meta, dict):
                        self.captured[name] = {
                            k: v.detach()
                            for k, v in meta.items()
                            if isinstance(v, torch.Tensor)
                        }
            else:
                if isinstance(output, torch.Tensor):
                    self.captured[name] = output.detach()

        return hook

    def clear(self):
        self.captured.clear()


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


def compute_layer_cosine_similarities(layer_outputs: dict) -> list:
    """Mean cosine similarity between consecutive TransformerBlock outputs.

    Args:
        layer_outputs: {module_name: tensor [B, S, D]}, sorted by name.

    Returns:
        List of floats (one per consecutive layer pair).
        Near 1.0 → block barely transforms the residual stream (passthrough).
        Near 0.0 → block applies a near-orthogonal transformation.
    """
    import torch.nn.functional as F

    names = sorted(layer_outputs.keys())
    sims = []
    with torch.no_grad():
        for i in range(len(names) - 1):
            h1 = (
                layer_outputs[names[i]]
                .reshape(-1, layer_outputs[names[i]].shape[-1])
                .float()
            )
            h2 = (
                layer_outputs[names[i + 1]]
                .reshape(-1, layer_outputs[names[i + 1]].shape[-1])
                .float()
            )
            sims.append(F.cosine_similarity(h1, h2, dim=-1).mean().item())
    return sims
