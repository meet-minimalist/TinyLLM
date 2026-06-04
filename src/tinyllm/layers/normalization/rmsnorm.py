import torch
import torch.nn as nn

from src.tinyllm.layers.registry import LAYER_REGISTRY


@LAYER_REGISTRY.register("rms")
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, **kwargs):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * (x / norm)
