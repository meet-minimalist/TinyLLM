"""
GPT model (GPT-2 style decoder-only transformer).
"""

import torch
import torch.nn as nn

from src.tinyllm.factory.registry import MODEL_REGISTRY
from src.tinyllm.models.layers.embeddings import LearnablePositionalEmbeddings
from src.tinyllm.models.layers.transformer_block import TransformerBlock


@MODEL_REGISTRY.register("gpt")
class GPTModel(nn.Module):
    """GPT-style decoder-only language model."""

    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.emb_dim = config.emb_dim
        self.max_seq_len = config.max_seq_len

        self.token_embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        self.pos_embedding = LearnablePositionalEmbeddings(
            config.max_seq_len, config.emb_dim
        )

        block_cfg = {
            "d_model": config.emb_dim,
            "num_heads": config.num_heads,
            "ff_multiplier": config.ff_multiplier,
            "drop_prob": config.drop_prob,
            "attention_type": "mha",
            "ffn_type": "standard",
            "normalization": "layer_norm",
            "act_fn": config.get("act_fn", "gelu"),
        }
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(block_cfg) for _ in range(config.num_blocks)]
        )

        self.final_norm = nn.LayerNorm(config.emb_dim)
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

        if config.get("tie_weights", False):
            self.lm_head.weight = self.token_embedding.weight

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
        x = self.token_embedding(input_ids)
        x = self.pos_embedding(x)

        seq_len = input_ids.shape[1]
        causal = self._causal_mask(seq_len, x.device)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2).to(torch.float32)
            mask = (1.0 - mask) * torch.finfo(torch.float32).min
            causal = causal + mask

        for block in self.transformer_blocks:
            x = block(x, causal)

        return self.lm_head(self.final_norm(x))

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )
        return torch.where(
            mask.unsqueeze(0).unsqueeze(0), torch.finfo(torch.float32).min, 0.0
        )
