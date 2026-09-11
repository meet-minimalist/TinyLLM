"""nGPT: the normalized Transformer (Loshchilov et al., arXiv:2410.01131).

Every vector that carries meaning — token embeddings, the rows of every weight
matrix, and the hidden state between blocks — is constrained to unit L2 norm
along the embedding dimension. The hidden state therefore never leaves the
surface of a d_model-dimensional unit hypersphere.

Two consequences shape the whole design:

1. A residual add would push the hidden state off the sphere, so the residual
   stream is replaced by a normalized interpolation toward the block output
   (``NGPTBlock``). The step size of that interpolation is learned per channel.

2. Normalization layers become redundant. If the input is already unit norm and
   the weight rows are unit norm, there is nothing left for RMSNorm to correct,
   so nGPT removes every norm layer including the final one. The constraint is
   maintained by re-normalizing the matrices after each optimizer step
   (``normalize_weights``) rather than by a layer in the forward pass.

Because every dot product is now between unit vectors, its magnitude collapses
to ~1/sqrt(d_model). The learnable scaling factors below (s_qk, s_u, s_v, s_z)
exist to restore a usable dynamic range wherever that matters.
"""

import math

import torch
import torch.nn as nn


def l2norm(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Project onto the unit hypersphere along ``dim``.

    Done in fp32 regardless of autocast: under bf16 the sum of squares of a
    512-dim vector loses enough precision that the result is visibly off the
    sphere, and that error compounds across 8 blocks.
    """
    dtype = x.dtype
    return torch.nn.functional.normalize(x.float(), p=2, dim=dim).to(dtype)


class NGPTScale(nn.Module):
    """A learnable scaling factor under nGPT's init/scale parameterization.

    The paper stores the parameter at ``scale`` and multiplies by
    ``init / scale`` in the forward pass, so the effective value starts at
    ``init`` while the stored value starts at ``scale``. Since an Adam step
    moves the stored value by roughly the learning rate regardless of gradient
    magnitude, the effective value moves by ``lr * (init / scale)``. That ratio
    is what decouples how fast these scalars move from the global learning rate
    used for the matrices — which is the entire point of carrying two numbers
    instead of one.
    """

    def __init__(self, dim: int, init: float, scale: float):
        super().__init__()
        self.init = float(init)
        self.scale = float(scale)
        self.weight = nn.Parameter(torch.full((dim,), float(scale)))

    def forward(self) -> torch.Tensor:
        return self.weight * (self.init / self.scale)

    def extra_repr(self) -> str:
        return (
            f"dim={self.weight.numel()}, init={self.init}, scale={self.scale}"
        )


class NGPTBlock(nn.Module):
    """One nGPT block: two normalized interpolations, no norm layers.

    The baseline pre-norm block computes ``h = h + f(Norm(h))``. nGPT computes::

        h <- Norm(h + alpha * (Norm(f(h)) - h))

    which is a linear interpolation from the current point on the sphere toward
    the block's (normalized) output, followed by a projection back onto the
    sphere — a cheap stand-in for a geodesic step. ``alpha`` is a learned vector
    of size d_model, one step size per channel, which the paper calls the eigen
    learning rate. ``abs`` keeps it a forward interpolation: a negative alpha
    would step away from the block output and is never what we want.

    Note there is no pre-norm on the input to attn/ffn. The input is already
    unit norm by construction, and the weight rows are too, so a norm layer
    would have nothing to do.
    """

    def __init__(self, config: dict):
        super().__init__()
        from src.tinyllm.layers.registry import LAYER_REGISTRY

        d_model = config["d_model"]
        drop_rate = config.get("drop_rate", config.get("drop_prob", 0.0))
        attn_cls = LAYER_REGISTRY.get(
            config.get("attention", config.get("attention_type", "gqa"))
        )
        ffn_cls = LAYER_REGISTRY.get(
            config.get("ffn", config.get("ffn_type", "gated"))
        )

        shared_kwargs = {
            k: v
            for k, v in config.items()
            if k
            not in (
                "d_model",
                "num_heads",
                "drop_rate",
                "drop_prob",
                "emb_dim",
                "ff_multiplier",
                "ffn_multiplier",
            )
        }
        shared_kwargs["ngpt"] = True

        self.attn = attn_cls(
            emb_dim=d_model,
            num_heads=config["num_heads"],
            drop_prob=drop_rate,
            **shared_kwargs,
        )
        self.ffn = ffn_cls(
            emb_dim=d_model,
            ff_multiplier=config.get(
                "ff_multiplier", config.get("ffn_multiplier", 4)
            ),
            drop_rate=drop_rate,
            **shared_kwargs,
        )

        # alpha_init ~ 1/n_layers per the paper (0.05 for their configs).
        alpha_init = config.get("ngpt_alpha_init", 0.05)
        alpha_scale = config.get("ngpt_alpha_scale", 1.0 / math.sqrt(d_model))
        self.attn_alpha = NGPTScale(d_model, alpha_init, alpha_scale)
        self.ffn_alpha = NGPTScale(d_model, alpha_init, alpha_scale)

    def forward(
        self,
        x,
        mask=None,
        cos=None,
        sin=None,
        cu_seqlens=None,
    ):
        attn_out, _ = self.attn(
            x, mask=mask, cos=cos, sin=sin, cu_seqlens=cu_seqlens
        )
        x = l2norm(x + self.attn_alpha().abs() * (l2norm(attn_out) - x))

        ffn_out = self.ffn(x)
        x = l2norm(x + self.ffn_alpha().abs() * (l2norm(ffn_out) - x))
        return x

    @torch.no_grad()
    def normalize_weights(self):
        self.attn.normalize_weights()
        self.ffn.normalize_weights()


@torch.no_grad()
def normalize_linear_(layer: nn.Linear, dim: int):
    """Unit-normalize a Linear's weight along the embedding axis.

    ``dim`` must be passed explicitly by the caller, which knows the layer's
    role. It cannot be inferred from the shape: with d_model=512 and 8 heads,
    both ``q_proj`` and ``o_proj`` are 512x512, yet the embedding axis is dim 1
    for the input projection and dim 0 for the output projection. Guessing gets
    one of them wrong, and the failure is silent — the hidden state still lands
    on the sphere because the block re-normalizes it, so the only symptom is a
    model that quietly trains worse.

    nn.Linear stores weight as ``[out_features, in_features]``:
      - projection INTO the block (in_features == d_model)  -> dim=1
      - projection OUT of the block (out_features == d_model) -> dim=0
    """
    w = layer.weight
    w.copy_(torch.nn.functional.normalize(w.float(), p=2, dim=dim).to(w.dtype))
