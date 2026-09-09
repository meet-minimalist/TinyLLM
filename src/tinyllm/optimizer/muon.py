import torch
import torch.nn as nn


@torch.no_grad()
def _zeroth_power_via_newtonschulz(G, steps=5):
    orig_shape = G.shape
    A = G.view(-1, G.shape[-1]) if G.dim() > 2 else G
    A = A / (A.norm() + 1e-8)
    transposed = False
    if A.size(0) > A.size(1):
        A = A.T.contiguous()
        transposed = True
    for _ in range(steps):
        B = A @ A.T
        A = 1.5 * A - 0.5 * B @ A
    if transposed:
        A = A.T.contiguous()
    return A.reshape(orig_shape).to(G.dtype)


class Muon(torch.optim.Optimizer):
    """
    Muon optimizer — applies Newton-Schulz to 2D parameters.

    Based on: https://kellerjordan.github.io/posts/muon/
    """

    def __init__(
        self,
        params,
        lr=0.0002,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        weight_decay=0.0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                if weight_decay != 0:
                    g = g + weight_decay * p

                state = self.state[p]
                if momentum > 0:
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g + momentum * buf
                    else:
                        g = buf

                if g.ndim >= 2:
                    g = _zeroth_power_via_newtonschulz(g, steps=ns_steps)
                    # Newton-Schulz returns singular values ~1 regardless of
                    # shape, so a wide matrix would otherwise take the same
                    # step as a tall one. Scale by the aspect ratio to keep the
                    # per-row update size consistent across shapes.
                    g = g * max(1.0, g.size(-2) / g.size(-1)) ** 0.5

                p.add_(g, alpha=-lr)

        return loss


def split_muon_adamw_params(
    model, muon_lr=0.0002, adamw_lr=0.001, weight_decay=0.1, momentum=0.95
):
    """
    Split model parameters into Muon and AdamW groups.

    Muon: 2D weight matrices (nn.Linear weights, no bias)
    AdamW: everything else (1D params, biases, embeddings, normalization weights)

    Embeddings and the LM head are excluded from Muon deliberately: their rows
    are per-token vectors rather than a transform between two feature spaces,
    so orthogonalizing their gradient is not meaningful. With tied weights the
    embedding tensor is also the classifier, which makes this doubly important.
    """
    muon_params = []
    adamw_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_embed_or_head = "embedding" in name or "lm_head" in name
        if p.ndim >= 2 and not is_embed_or_head:
            muon_params.append(p)
        else:
            adamw_params.append(p)

    return [
        {
            "params": muon_params,
            "lr": muon_lr,
            "momentum": momentum,
            "weight_decay": 0.0,
        },
        {"params": adamw_params, "lr": adamw_lr, "weight_decay": weight_decay},
    ]


def build_muon_adamw_optimizer(
    model,
    muon_lr=0.0002,
    adamw_lr=0.001,
    weight_decay=0.1,
    momentum=0.95,
    adamw_betas=(0.9, 0.95),
    use_8bit=False,
):
    param_groups = split_muon_adamw_params(
        model, muon_lr, adamw_lr, weight_decay, momentum
    )
    muon_group = param_groups[0]
    adamw_group = param_groups[1]

    muon_opt = Muon(
        muon_group["params"],
        lr=muon_group["lr"],
        momentum=muon_group["momentum"],
        weight_decay=muon_group.get("weight_decay", 0.0),
    )

    if use_8bit:
        try:
            import bitsandbytes as bnb

            adamw_opt = bnb.optim.AdamW8bit(
                adamw_group["params"],
                lr=adamw_group["lr"],
                betas=adamw_betas,
                weight_decay=adamw_group["weight_decay"],
            )
        except ImportError:
            from src.tinyllm.logger.logger_utils import logger

            logger.warning(
                "bitsandbytes not installed. Falling back to standard AdamW. "
                "Install with: pip install bitsandbytes"
            )
            adamw_opt = torch.optim.AdamW(
                adamw_group["params"],
                lr=adamw_group["lr"],
                betas=adamw_betas,
                weight_decay=adamw_group["weight_decay"],
                fused=True,
            )
    else:
        adamw_opt = torch.optim.AdamW(
            adamw_group["params"],
            lr=adamw_group["lr"],
            betas=adamw_betas,
            weight_decay=adamw_group["weight_decay"],
            fused=True,
        )

    return _CombinedOptimizer(muon_opt, adamw_opt)


class _CombinedOptimizer(torch.optim.Optimizer):
    """
    Combines Muon (2D weights) and AdamW (1D/biases/embeddings/norms).
    Subclasses torch.optim.Optimizer so it passes isinstance() checks
    required by lr_schedulers (LambdaLR, CosineWarmup, etc.).

    The param_groups reference the SAME dict objects as the internal
    optimizers, so lr_scheduler modifications propagate correctly.
    """

    def __init__(self, muon_opt, adamw_opt):
        self.muon = muon_opt
        self.adamw = adamw_opt
        # super().__init__ validates param groups and sets up self.state
        super().__init__(muon_opt.param_groups + adamw_opt.param_groups, {})
        # Use ORIGINAL param_group dicts so lr changes propagate to
        # the internal optimizers (add_param_group shallow-copies)
        self.param_groups = muon_opt.param_groups + adamw_opt.param_groups

    def step(self, closure=None):
        self.muon.step(closure)
        self.adamw.step(closure)

    def zero_grad(self, set_to_none=True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "muon": self.muon.state_dict(),
            "adamw": self.adamw.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])
