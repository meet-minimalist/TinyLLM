import torch.nn as nn

from src.tinyllm.layers.registry import LAYER_REGISTRY


@LAYER_REGISTRY.register("rope_only")
class RopeOnlyEmbedding(nn.Module):
    """
    No-op position embedding layer.

    Used with RoPE-based models where positional information is injected
    exclusively through rotary transformations on Q and K tensors inside
    each attention layer. No position information is added at the embedding level.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        return x
