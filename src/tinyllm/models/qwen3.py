"""
Qwen3 model using shared layer components.
"""

import torch
import torch.nn as nn

from src.tinyllm.factory.registry import MODEL_REGISTRY
from src.tinyllm.models.layers.rope import RotaryPositionalEmbedding
from src.tinyllm.models.layers.normalization import RMSNorm
from src.tinyllm.models.layers.transformer_block import TransformerBlock


@MODEL_REGISTRY.register("qwen3")
class Qwen3(nn.Module):
    """Qwen3-style language model with GQA and RoPE."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        d_model = config["d_model"]
        vocab_size = config["vocab_size"]
        num_layers = config["num_layers"]

        self.embedding = nn.Embedding(vocab_size, d_model)

        head_dim = d_model // config["num_heads"]
        self.rope = RotaryPositionalEmbedding(
            head_dim, max_seq_len=config.get("max_seq_len", 2048)
        )

        block_cfg = {
            **config,
            "attention_type": "gqa",
            "ffn_type": "gated",
            "normalization": "rms",
        }
        self.transformer_layers = nn.ModuleList(
            [TransformerBlock(block_cfg) for _ in range(num_layers)]
        )

        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if config.get("tie_word_embeddings", False):
            self.lm_head.weight = self.embedding.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, input_ids: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        _, seq_len = input_ids.shape
        x = self.embedding(input_ids)

        cos, sin = self.rope(x, seq_len)

        if mask is None:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1,
            )

        for layer in self.transformer_layers:
            x = layer(x, mask, cos, sin)

        return self.lm_head(self.final_norm(x))
