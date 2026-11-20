import numpy as np

from mortgage_elasticity.model import beta_from_uplift, lag, multiplier, u_from_beta


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
