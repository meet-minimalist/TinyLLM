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

                p.add_(g, alpha=-lr)

        return loss


def split_muon_adamw_params(
    model, muon_lr=0.0002, adamw_lr=0.001, weight_decay=0.1, momentum=0.95
):
    """
    Split model parameters into Muon and AdamW groups.

    Muon: 2D weight matrices (nn.Linear weights, no bias)
    AdamW: everything else (1D params, biases, embeddings, normalization weights)
    """
    muon_params = []
    adamw_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
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

    adamw_opt = torch.optim.AdamW(
        adamw_group["params"],
        lr=adamw_group["lr"],
        betas=adamw_betas,
        weight_decay=adamw_group["weight_decay"],
        fused=True,
    )

    return _CombinedOptimizer(muon_opt, adamw_opt)


class _CombinedOptimizer:
    def __init__(self, muon_opt, adamw_opt):
        self.muon = muon_opt
        self.adamw = adamw_opt

    @property
    def param_groups(self):
        return self.muon.param_groups + self.adamw.param_groups

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
