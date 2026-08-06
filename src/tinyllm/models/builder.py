import torch
import torch.nn as nn

from src.tinyllm.layers.registry import LAYER_REGISTRY
from src.tinyllm.layers.transformer_block import TransformerBlock
from src.tinyllm.models.base import BaseLLM
from src.tinyllm.models.layers.rope import RotaryPositionalEmbedding
from src.tinyllm.factory.registry import MODEL_REGISTRY


def _doc_position_ids(
    cu_seqlens: torch.Tensor, seq_len: int, device: torch.device
) -> torch.Tensor:
    """Per-document RoPE positions for a varlen-packed row.

    Positions run ``0..len-1`` within each document and reset to 0 at every
    document boundary in ``cu_seqlens``, so a packed row containing many
    documents never uses an absolute position larger than the longest
    document (which the packer caps at ``max_seq_len``). Fully vectorized, and
    correct even when the packer emits a zero-length trailing document (a
    duplicated ``cu_seqlens`` value from clamping the +1 target shift).
    """
    positions = torch.arange(seq_len, device=device, dtype=torch.long)
    cu = cu_seqlens.to(device=device, dtype=torch.long)
    # doc_id[t] = index of the rightmost boundary <= t. searchsorted(right=True)
    # naturally skips zero-length docs (duplicate boundaries), so each token maps
    # to the real document that contains it.
    doc_id = torch.searchsorted(cu, positions, right=True) - 1
    doc_id = doc_id.clamp_(min=0)
    start_of_doc = cu[doc_id]  # [seq_len]
    return positions - start_of_doc


def _first_key(cfg, *keys, default=None):
    for k in keys:
        if isinstance(cfg, dict):
            v = cfg.get(k)
        else:
            v = getattr(cfg, k, None)
        if v is not None:
            return v
    return default


@MODEL_REGISTRY.register("dynamic")
class DynamicModel(BaseLLM):
    def __init__(self, config):
        super().__init__(config)

        cfg = config if isinstance(config, dict) else config.to_dict()
        d_model = cfg["d_model"]
        vocab_size = cfg["vocab_size"]
        max_seq_len = cfg["max_seq_len"]
        blocks_cfg = _first_key(cfg, "blocks", default={})
        num_heads = _first_key(blocks_cfg, "num_heads", default=4)

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        embed_type = cfg.get("embedding", "learned_pe")
        embed_cls = LAYER_REGISTRY.get(embed_type)
        self.position_embedding = embed_cls(
            max_seq_len=max_seq_len, emb_dim=d_model
        )

        if embed_type == "rope_only":
            head_dim = d_model // num_heads
            self.rope = RotaryPositionalEmbedding(
                head_dim=head_dim,
                max_seq_len=max_seq_len,
                scaling_type=cfg.get("rope_scaling_type"),
                scaling_factor=cfg.get("rope_scaling_factor", 1.0),
                original_max_seq_len=cfg.get("rope_original_max_seq_len"),
                beta_fast=cfg.get("rope_beta_fast", 32.0),
                beta_slow=cfg.get("rope_beta_slow", 1.0),
            )
        else:
            self.rope = None

        ff_mult = _first_key(
            blocks_cfg, "ff_multiplier", "ffn_multiplier", default=4
        )
        block_config = {
            "d_model": d_model,
            "num_heads": _first_key(blocks_cfg, "num_heads", default=4),
            "ff_multiplier": ff_mult,
            "drop_prob": _first_key(blocks_cfg, "drop_prob", default=0.0),
            "attention": _first_key(blocks_cfg, "attention", default="mha"),
            "ffn": _first_key(blocks_cfg, "ffn", default="standard"),
            "norm": _first_key(blocks_cfg, "norm", default="layer_norm"),
            "act_fn": _first_key(blocks_cfg, "act_fn", default="gelu"),
        }
        for k, v in blocks_cfg.items():
            if k not in (
                "count",
                "attention",
                "ffn",
                "norm",
                "num_heads",
                "ff_multiplier",
                "ffn_multiplier",
                "drop_prob",
                "drop_rate",
                "act_fn",
            ):
                block_config[k] = v

        num_blocks = _first_key(blocks_cfg, "count", default=4)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(block_config) for _ in range(num_blocks)]
        )

        head_cfg = _first_key(cfg, "head", default={})
        norm_type = _first_key(head_cfg, "norm", default="layer_norm")
        norm_cls = LAYER_REGISTRY.get(norm_type)
        self.final_norm = norm_cls(d_model)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        tie = _first_key(head_cfg, "tie_weights", default=False)
        if tie:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def forward(
        self, input_ids, mask=None, cu_seqlens=None, return_hidden=False
    ):
        x = self.token_embedding(input_ids)
        x = self.position_embedding(x)

        if self.rope is not None:
            # In varlen packing, reset RoPE positions at each document boundary
            # so positions stay within [0, max_seq_len) no matter how many docs
            # are packed per row. Without cu_seqlens (fixed_batch), fall back to
            # contiguous 0..S-1 positions.
            position_ids = (
                _doc_position_ids(cu_seqlens, x.shape[1], x.device)
                if cu_seqlens is not None
                else None
            )
            cos, sin = self.rope(
                x, seq_len=x.shape[1], position_ids=position_ids
            )
        else:
            cos, sin = None, None

        for block in self.transformer_blocks:
            x = block(x, mask=mask, cos=cos, sin=sin, cu_seqlens=cu_seqlens)

        hidden = self.final_norm(x)
        if return_hidden:
            return hidden  # [B, S, d_model] — caller handles lm_head + loss
        return self.lm_head(hidden)
