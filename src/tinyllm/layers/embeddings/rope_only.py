import torch.nn as nn

from src.tinyllm.layers.registry import LAYER_REGISTRY


@LAYER_REGISTRY.register("rope_only")
class RopeOnlyEmbedding(nn.Module):
    def forward(self, x):
        return x
