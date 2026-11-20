import numpy as np
import pytest

from mortgage_elasticity.model import (
    beta_from_uplift,
    lag,
    lag_kernel,
    multiplier,
    path_multiplier,
    step_response,
    u_from_beta,
)


def test_round_trip_returns_to_baseline():
    b = beta_from_uplift(0.08, 0.12)
    x = np.cumsum([-5, 5] * 6)
    assert np.isclose(multiplier(x[-1], 0.12, b), 1.0)


def test_matches_linear_uplift_at_margin():
    s0, u = 0.12, 0.08
    b = beta_from_uplift(u, s0)
    h = 1e-5
    slope = (np.log(multiplier(h, s0, b)) - np.log(multiplier(-h, s0, b))) / (2 * h)
    assert np.isclose(slope, -u, rtol=1e-4)
    assert np.isclose(u_from_beta(b, s0), u)


def test_ceiling_is_whole_segment():
    s0 = 0.12
    assert multiplier(-1000, s0, beta_from_uplift(0.08, s0)) <= 1 / s0 + 1e-9


def test_lag_converges_and_is_path_independent():
    path = np.r_[np.zeros(5), np.full(60, -10.0)]
    assert np.isclose(lag(path, 2.0)[-1], -10.0, atol=1e-6)


@pytest.mark.parametrize("h", [0.5, 1.0, 2.0, 5.0, 12.0])
def test_lag_kernel_equals_the_recursion(h):
    from mortgage_elasticity.bayes import lag_kernel_pt
    v = 3.0 + np.cumsum(np.random.default_rng(0).choice([0, 0, 0, -10, 5, 15], size=40))
    want = lag(v, h)
    assert np.allclose(v[0] + lag_kernel(len(v), h) @ (v - v[0]), want)
    W = lag_kernel_pt(len(v), 1 - 2.0 ** (-1 / h)).eval()  # the kernel inside the PyMC graph
    assert np.allclose(v[0] + W @ (v - v[0]), want)


def test_path_multiplier_phases_in_towards_the_long_run():
    s, b, dx = 0.10, beta_from_uplift(0.03, 0.10), -10.0
    weekly, avg = path_multiplier(dx, s, b, half_life=2.0, horizon=12)
    assert np.all(np.diff(weekly) > 0) and weekly[0] > 1
    assert weekly[-1] < multiplier(dx, s, b) and np.isclose(avg, weekly.mean())
    assert np.isclose(step_response(2, 2.0), 0.5)  # half felt after one half-life
    # the first 12 weeks see roughly 80% of a move at a two-week half-life
    assert 0.75 < step_response(np.arange(1, 13), 2.0).mean() < 0.85
