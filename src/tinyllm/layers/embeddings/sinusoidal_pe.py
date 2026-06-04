import math
import torch
import torch.nn as nn

from src.tinyllm.layers.registry import LAYER_REGISTRY


@LAYER_REGISTRY.register("sinusoidal")
class SinusoidalPositionalEmbeddings(nn.Module):
    def __init__(self, max_seq_len: int, emb_dim: int, **kwargs):
        super().__init__()
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, emb_dim, 2) * (-math.log(10000.0) / emb_dim)
        )
        pe = torch.zeros(1, max_seq_len, emb_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        return x + self.pe[:, :seq_len, :]
