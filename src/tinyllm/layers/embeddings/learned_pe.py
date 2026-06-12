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
        if seq_len > self.max_seq_len:
            raise RuntimeError(
                f"Sequence length {seq_len} exceeds position embedding "
                f"max_seq_len={self.max_seq_len}. "
                "Check that train_config.packed_tokens and model_config.max_seq_len "
                "are consistent."
            )
        return x + self.pos_emb[:, :seq_len, :]
