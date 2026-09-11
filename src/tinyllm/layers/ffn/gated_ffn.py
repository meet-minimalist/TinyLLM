import math

import torch
import torch.nn as nn

from src.tinyllm.layers.registry import LAYER_REGISTRY

_ACT_DICT = {"swish": nn.SiLU(), "gelu": nn.GELU()}


@LAYER_REGISTRY.register("gated")
class GatedFFN(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        ff_multiplier: int = 4,
        drop_rate: float = 0.0,
        act_fn: str = "swish",
        ngpt: bool = False,
        **kwargs
    ):
        super().__init__()
        # int() so fractional multipliers (e.g. SwiGLU's ~2/3*4 = 2.67) yield a
        # valid integer hidden size; nn.Linear rejects float dimensions.
        hidden = int(emb_dim * ff_multiplier)
        self.gate = nn.Linear(emb_dim, hidden, bias=False)
        self.up = nn.Linear(emb_dim, hidden, bias=False)
        self.down = nn.Linear(hidden, emb_dim, bias=False)
        self.dropout = nn.Dropout(drop_rate)
        self.act = _ACT_DICT.get(act_fn, nn.SiLU())
        self.ngpt = ngpt
        self.emb_dim = emb_dim

        if ngpt:
            from src.tinyllm.layers.ngpt import NGPTScale

            # With unit-norm weight rows and a unit-norm input, each projection
            # output is a dot product of unit vectors, so it sits around
            # 1/sqrt(d_model) — deep in SiLU's linear region, where the
            # non-linearity does nothing. s_v carries an extra sqrt(d_model)
            # to put the gate back into the part of SiLU that actually bends.
            self.u_scale = NGPTScale(hidden, 1.0, 1.0)
            self.v_scale = NGPTScale(hidden, 1.0, 1.0)
            self.v_extra = math.sqrt(emb_dim)
        else:
            self.u_scale = None
            self.v_scale = None

    def forward(self, x):
        if self.ngpt:
            u = self.up(x) * self.u_scale()
            v = self.gate(x) * self.v_scale() * self.v_extra
            return self.dropout(self.down(u * self.act(v)))
        return self.dropout(self.down(self.act(self.gate(x)) * self.up(x)))

    @torch.no_grad()
    def normalize_weights(self):
        from src.tinyllm.layers.ngpt import normalize_linear_

        # gate/up read from the residual stream, down writes back into it.
        for proj in (self.gate, self.up):
            normalize_linear_(proj, dim=1)
        normalize_linear_(self.down, dim=0)
