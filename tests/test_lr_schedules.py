import math

import pytest
import torch

from src.tinyllm.utils.lr_schedules import build_lambda_schedule, power_lambda, wsd_lambda


def test_wsd_shape():
    fn = wsd_lambda(100, num_warmup_steps=10, decay_steps=20)
    assert fn(0) == pytest.approx(0.1)
    assert fn(9) == pytest.approx(1.0)
    assert fn(50) == 1.0
    assert fn(79) == 1.0
    assert fn(80) == pytest.approx(1.0)
    assert fn(90) == pytest.approx(0.5)
    assert fn(100) == pytest.approx(0.0)


def test_wsd_min_ratio_and_cosine():
    fn = wsd_lambda(100, 0, 20, decay_shape="cosine", min_lr_ratio=0.1)
    assert fn(90) == pytest.approx(0.1 + 0.9 * 0.5)
    assert fn(100) == pytest.approx(0.1)


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
        return opt, build_lambda_schedule(name, opt, 100, 10, {"decay_steps": 0.2})

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
