import torch.nn as nn

from src.tinyllm.layers.registry import LAYER_REGISTRY


@LAYER_REGISTRY.register("layer_norm")
class LayerNormWrapper(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        return self.norm(x)
