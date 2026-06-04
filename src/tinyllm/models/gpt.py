import torch.nn as nn

from src.tinyllm.factory.registry import MODEL_REGISTRY
from src.tinyllm.layers.registry import LAYER_REGISTRY
from src.tinyllm.layers.transformer_block import TransformerBlock
from src.tinyllm.models.base import BaseLLM


@MODEL_REGISTRY.register("gpt")
class GPTModel(BaseLLM):
    def __init__(self, config):
        super().__init__(config)
        d_model = config.emb_dim
        vocab_size = config.vocab_size
        max_seq_len = config.max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        embed_cls = LAYER_REGISTRY.get("learned_pe")
        self.pos_embedding = embed_cls(max_seq_len=max_seq_len, emb_dim=d_model)

        block_cfg = {
            "d_model": d_model,
            "num_heads": config.num_heads,
            "ff_multiplier": config.get("ff_multiplier", 4),
            "drop_prob": config.get("drop_prob", 0.0),
            "attention": "mha",
            "ffn": "standard",
            "norm": "layer_norm",
            "act_fn": config.get("act_fn", "gelu"),
        }
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(block_cfg) for _ in range(config.num_blocks)]
        )

        norm_cls = LAYER_REGISTRY.get("layer_norm")
        self.final_norm = norm_cls(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if config.get("tie_weights", False):
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def forward(self, input_ids, mask=None):
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
