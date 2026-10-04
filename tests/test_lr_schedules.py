import math

import pytest
import torch

from src.tinyllm.factory.factory import lr_scheduler_factory
from src.tinyllm.utils.lr_schedules import power_lambda


def _lrs(name, steps, warmup, cfg):
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1.0)
    sched = lr_scheduler_factory(name, opt, steps, warmup, cfg)
    out = []
    for _ in range(steps + 1):
        out.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    return out


def test_wsd_from_factory_shape():
    # transformers' get_wsd_schedule; fractions resolve against num_training_steps.
    lr = _lrs("wsd", 100, 10, {"decay_steps": 0.2, "decay_type": "linear"})
    assert lr[5] == pytest.approx(0.5)
    assert lr[50] == 1.0 and lr[79] == 1.0
    assert lr[90] == pytest.approx(0.5)
    assert lr[100] == pytest.approx(0.0)


def test_power_matches_rigel_form():
    # Rigel: lr = min(0.01, 1.819 * s^-0.51) -> constant until s0, then s^-0.51.
    eta_max, a, b = 0.01, 1.819, 0.51
    s0 = (a / eta_max) ** (1 / b)
    fn = power_lambda(10**6, 5000, power_start_step=int(s0), exponent=b, decay_steps=0)
    for s in (6000, 20000, 100000, 500000):
        expected = min(eta_max, a * s**-b)
        assert eta_max * fn(s) == pytest.approx(expected, rel=1e-3)


def test_power_final_linear_decay_to_zero():
    fn = power_lambda(1000, 10, power_start_step=100, exponent=0.5, decay_steps=100)
    start = (900 / 100) ** -0.5
    assert fn(900) == pytest.approx(start)
    assert fn(950) == pytest.approx(start / 2)
    assert fn(1000) == pytest.approx(0.0)


@pytest.mark.parametrize("name", ["wsd", "power"])
def test_schedule_state_resumes(name):
    def make():
        p = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.SGD([p], lr=1.0)
        return opt, lr_scheduler_factory(name, opt, 100, 10, {"decay_steps": 0.2})

    opt, sched = make()
    for _ in range(37):
        opt.step()
        sched.step()
    state = sched.state_dict()
    opt2, sched2 = make()
    sched2.load_state_dict(state)
    opt2.param_groups[0]["lr"] = opt.param_groups[0]["lr"]
    for _ in range(20):
        opt.step()
        sched.step()
        opt2.step()
        sched2.step()
        assert math.isclose(opt.param_groups[0]["lr"], opt2.param_groups[0]["lr"])
