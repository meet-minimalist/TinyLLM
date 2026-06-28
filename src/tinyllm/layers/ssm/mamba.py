"""
Mamba SSM (State Space Model) block — the selective scan architecture.

Based on:
  Mamba: Linear-Time Sequence Modeling with Selective State Spaces
  https://arxiv.org/abs/2312.00752

This implements a simplified Mamba block that can be used as a drop-in
replacement for attention in a transformer block (via `attention: "mamba"`).
Uses a selective scan over the hidden state with discretized SSM parameters.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tinyllm.layers.registry import LAYER_REGISTRY


@LAYER_REGISTRY.register("mamba")
class MambaSSM(nn.Module):
    """
    Simplified Mamba SSM block.

    Returns (output, metadata) to match the attention module interface.
    """

    def __init__(
        self,
        emb_dim: int,
        num_heads: int = 1,  # ignored — Mamba uses a single state
        drop_prob: float = 0.0,
        d_state: int = 16,   # SSM state dimension
        d_conv: int = 4,     # local convolution width
        expand_factor: int = 2,  # expansion factor for the inner dimension
        **kwargs,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand_factor = expand_factor
        inner_dim = int(expand_factor * emb_dim)

        # Input projection: x -> (z, x')
        self.in_proj = nn.Linear(emb_dim, inner_dim * 2, bias=False)

        # Convolution for selective scanning
        self.conv1d = nn.Conv1d(
            in_channels=inner_dim,
            out_channels=inner_dim,
            bias=True,
            kernel_size=d_conv,
            groups=inner_dim,
            padding=d_conv - 1,
        )

        # SSM parameters (input-dependent via linear projections)
        self.x_proj = nn.Linear(inner_dim, d_state + d_state + inner_dim, bias=False)
        self.dt_proj = nn.Linear(d_state, inner_dim, bias=True)

        # Output projection
        self.out_proj = nn.Linear(inner_dim, emb_dim, bias=False)
        self.dropout = nn.Dropout(drop_prob)

        # Learnable dt bias
        dt_init = torch.rand(inner_dim) * 0.1
        self.dt_bias = nn.Parameter(dt_init)

        # A parameter (log-space for stability)
        A = torch.arange(1, d_state + 1, dtype=torch.float).repeat(inner_dim, 1)
        self.A_log = nn.Parameter(torch.log(A))

        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(inner_dim))

    def forward(
        self,
        x: torch.Tensor,
        mask=None,
        cos=None,
        sin=None,
        cu_seqlens=None,
    ) -> Tuple[torch.Tensor, dict]:
        """Mamba forward. Returns (output, metadata)."""
        B, L, D = x.shape
        inner_dim = int(self.expand_factor * D)

        # ---- Input projection ----
        z_x = self.in_proj(x)  # [B, L, inner_dim * 2]
        z, x_in = z_x.chunk(2, dim=-1)
        z = F.silu(z)

        # ---- 1D Convolution (causal) ----
        x_conv = x_in.transpose(-1, -2)  # [B, inner_dim, L]
        x_conv = self.conv1d(x_conv)[..., :L]  # remove padding
        x_conv = F.silu(x_conv)

        # ---- SSM parameters ----
        # x_proj -> (delta, A, B, C) where delta is input-dependent
        dt_xb = self.x_proj(x_conv.transpose(-1, -2))  # [B, L, d_state*2 + inner_dim]
        dt, B_param, C_param = dt_xb.split(
            [self.d_state, self.d_state, inner_dim], dim=-1
        )

        # Delta: softplus
        dt = F.softplus(dt + self.dt_bias)
        dt = self.dt_proj(dt)  # [B, L, inner_dim]

        # A: negative exponential
        A = -torch.exp(self.A_log.float())  # [inner_dim, d_state]

        # ---- Selective scan (simplified) ----
        # Discretize: A_bar = exp(dt * A), B_bar = dt * B
        # Simplified scanning: cumulative approximation
        deltaA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        deltaB = dt.unsqueeze(-1) * B_param.unsqueeze(-2)  # [B, L, inner_dim, d_state]
        Bx = (deltaB * x_conv.transpose(-1, -2).unsqueeze(-1)).sum(dim=-2)

        # Recurrence: h_t = deltaA_t * h_{t-1} + Bx_t
        h = torch.zeros(B, inner_dim, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h = deltaA[:, t] * h + Bx[:, t]  # element-wise
            y = (h * C_param[:, t].unsqueeze(-1)).sum(dim=-1)
            ys.append(y)

        y = torch.stack(ys, dim=1)  # [B, L, inner_dim]

        # Skip connection
        y = y + F.silu(x_in) * self.D.unsqueeze(0).unsqueeze(0)

        # ---- Gating ----
        out = y * z
        out_proj = self.out_proj(out)

        metadata = {
            "q": None, "k": None, "v": None,
            "attn_logits": None, "attn_weights": None,
            "qkv_out": out, "o_proj_out": out_proj,
        }
        return out_proj, metadata
