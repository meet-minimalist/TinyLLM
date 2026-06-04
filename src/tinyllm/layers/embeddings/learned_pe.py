import torch
import torch.nn as nn

from src.tinyllm.layers.registry import LAYER_REGISTRY


@LAYER_REGISTRY.register("learned_pe")
class LearnablePositionalEmbeddings(nn.Module):
    def __init__(self, max_seq_len: int, emb_dim: int, **kwargs):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, emb_dim) * 0.02)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        assert (
            seq_len <= self.max_seq_len
        ), f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
        return x + self.pos_emb[:, :seq_len, :]
