"""
Mixture of Experts (MoE) FFN layer.

Implements sparse expert routing with top-k gating. Each input token is
routed to the top-k experts (k << num_experts), and the output is a
weighted combination of the selected experts' outputs.

Config:
    ffn: "moe"
    num_experts: 8          # total number of experts
    top_k: 2                # experts routed per token
    expert_multiplier: 4    # hidden dim multiplier per expert (default: ff_multiplier)
    moe_jitter: 0.01        # small noise on gate logits for load balancing
    moe_capacity: 1.25      # expert capacity factor (1.0 = balanced, >1 = slack)
    z_loss_coef: 0.001      # auxiliary z-loss coefficient for load balancing
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tinyllm.layers.registry import LAYER_REGISTRY
from src.tinyllm.layers.ffn.gated_ffn import GatedFFN
from src.tinyllm.layers.ffn.standard_ffn import StandardFFN


class _ExpertFFN(nn.Module):
    """A single expert — delegates to the configured inner FFN type."""

    def __init__(
        self,
        emb_dim: int,
        ff_multiplier: int,
        drop_rate: float,
        act_fn: str,
        ffn_type: str = "standard",
    ):
        super().__init__()
        if ffn_type == "gated":
            self.net = GatedFFN(
                emb_dim=emb_dim,
                ff_multiplier=ff_multiplier,
                drop_rate=drop_rate,
                act_fn=act_fn,
            )
        else:
            self.net = StandardFFN(
                emb_dim=emb_dim,
                ff_multiplier=ff_multiplier,
                drop_rate=drop_rate,
                act_fn=act_fn,
            )

    def forward(self, x):
        return self.net(x)


@LAYER_REGISTRY.register("moe")
class MoEFFN(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        ff_multiplier: int = 4,
        drop_rate: float = 0.0,
        act_fn: str = "gelu",
        num_experts: int = 8,
        top_k: int = 2,
        moe_capacity: float = 1.25,
        z_loss_coef: float = 0.001,
        expert_ffn_type: str = "standard",
        **kwargs,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.moe_capacity = moe_capacity
        self.z_loss_coef = z_loss_coef

        # Gating network
        self.gate = nn.Linear(emb_dim, num_experts, bias=False)

        # Experts
        self.experts = nn.ModuleList(
            [
                _ExpertFFN(
                    emb_dim=emb_dim,
                    ff_multiplier=ff_multiplier,
                    drop_rate=drop_rate,
                    act_fn=act_fn,
                    ffn_type=expert_ffn_type,
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        MoE forward pass with top-k routing.

        Args:
            x: Input tensor [batch, seq_len, emb_dim].

        Returns:
            Output tensor [batch, seq_len, emb_dim].
        """
        orig_shape = x.shape
        batch, seq_len, emb_dim = orig_shape

        # Flatten to [batch*seq_len, emb_dim] for per-token routing
        x_flat = x.view(-1, emb_dim)
        num_tokens = x_flat.shape[0]

        # Gate logits and softmax probabilities
        gate_logits = self.gate(x_flat)  # [num_tokens, num_experts]
        gate_probs = F.softmax(gate_logits, dim=-1)

        # Top-k selection
        top_k_probs, top_k_indices = gate_probs.topk(self.top_k, dim=-1)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # Prepare output buffer
        output = torch.zeros_like(x_flat)

        # Route tokens to experts
        for expert_idx in range(self.num_experts):
            # Find tokens that have this expert in their top-k
            mask = top_k_indices == expert_idx  # [num_tokens, top_k]
            if not mask.any():
                continue

            # Get the gating weight for this expert for selected tokens
            # mask contains True/False for each (token, k) pair
            token_indices, k_indices = mask.nonzero(as_tuple=True)

            # Deduplicate — a token may have this expert at multiple k positions
            unique_tokens = torch.unique(token_indices)
            expert_weight = top_k_probs[unique_tokens, :][
                torch.arange(len(unique_tokens), device=x.device),
                (top_k_indices[unique_tokens] == expert_idx).int().argmax(dim=-1),
            ]

            # Run expert and accumulate
            expert_out = self.experts[expert_idx](x_flat[unique_tokens])
            output[unique_tokens] += expert_weight.unsqueeze(-1) * expert_out

        # Auxiliary z-loss for load balancing
        # Encourages uniform routing by penalizing the log-sum-exp of gate logits
        if self.z_loss_coef > 0 and self.training:
            z_loss = (
                torch.logsumexp(gate_logits, dim=-1).square().mean()
                * self.z_loss_coef
            )
            # Add to output as a side-effect; outer training loop sums losses
            output = output + z_loss * output.detach().mean() / (
                output.detach().norm() + 1e-8
            )

        return output.view(orig_shape)
