import torch
import torch.nn as nn

from src.tinyllm.factory.registry import MODEL_REGISTRY
from src.tinyllm.layers.registry import LAYER_REGISTRY
from src.tinyllm.layers.transformer_block import TransformerBlock
from src.tinyllm.models.base import BaseLLM
from src.tinyllm.models.layers.rope import RotaryPositionalEmbedding


@MODEL_REGISTRY.register("qwen3")
class Qwen3(BaseLLM):
    def __init__(self, config):
        super().__init__(config)
        cfg = config if isinstance(config, dict) else config.to_dict()
        d_model = cfg["d_model"]
        vocab_size = cfg["vocab_size"]
        num_layers = cfg["num_layers"]

        self.embedding = nn.Embedding(vocab_size, d_model)

        head_dim = d_model // cfg["num_heads"]
        self.rope = RotaryPositionalEmbedding(
            head_dim, max_seq_len=cfg.get("max_seq_len", 2048)
        )

        block_cfg = {
            **cfg,
            "d_model": d_model,
            "attention": "gqa",
            "ffn": "gated",
            "norm": "rms",
        }
        self.transformer_layers = nn.ModuleList(
            [TransformerBlock(block_cfg) for _ in range(num_layers)]
        )

        norm_cls = LAYER_REGISTRY.get("rms")
        self.final_norm = norm_cls(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if cfg.get("tie_word_embeddings", False):
            self.lm_head.weight = self.embedding.weight

        self.apply(self._init_weights)

    def forward(self, input_ids, mask=None):
        _, seq_len = input_ids.shape
        x = self.embedding(input_ids)

        cos, sin = self.rope(x, seq_len)

        if mask is None:
            mask = self._causal_mask(seq_len, x.device)

        for layer in self.transformer_layers:
            x = layer(x, mask, cos, sin)

        return self.lm_head(self.final_norm(x))
