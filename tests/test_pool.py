"""Prior -> posterior per segment, and pooled across segments."""

import numpy as np
import pandas as pd

from mortgage_elasticity.pool import pooled_betas, posterior_betas

S0 = pd.Series({"A": 0.10, "B": 0.20}, name="s0")
NO_EVENTS = pd.DataFrame(columns=["segment", "clean", "beta_e", "beta_e_se"])


def _ev(rows):
    return pd.DataFrame(rows, columns=["segment", "clean", "beta_e", "beta_e_se"])


def test_posterior_without_events_is_prior():
    p = posterior_betas(S0, NO_EVENTS, 0.08, 0.3).set_index("segment")
    assert np.allclose(p["beta_post"], p["beta_prior"]) and np.allclose(p["prior_weight"], 1)


def test_posterior_normal_normal_and_unclean_ignored():
    b0 = 0.08 / 0.9
    ev = _ev([("A", True, 2 * b0, 0.3 * b0), ("A", False, 10 * b0, 0.01 * b0)])
    p = posterior_betas(S0, ev, 0.08, 0.3).set_index("segment")
    assert np.isclose(p.loc["A", "beta_post"], 1.5 * b0)  # se = sd0 -> midpoint
    assert np.isclose(p.loc["A", "prior_weight"], 0.5) and p.loc["A", "n_clean_events"] == 1


def test_pooled_without_events_is_exactly_the_prior():
    p = pooled_betas(S0, NO_EVENTS, 0.08, 0.3, 0.15).set_index("segment")
    b0 = 0.08 / (1 - S0)
    assert np.allclose(p["beta_pool"], b0) and np.allclose(p["beta_pool_sd"], 0.3 * b0)
    assert np.allclose(p["own_weight"], 0) and np.isclose(p["shared_scale"].iloc[0], 1)


def test_events_in_one_segment_move_the_other():
    b0a = 0.08 / 0.9
    ev = _ev([("A", True, 1.5 * b0a, 0.1 * b0a), ("A", False, 0.1 * b0a, 0.1 * b0a)])
    p = pooled_betas(S0, ev, 0.08, 0.3, 0.15).set_index("segment")
    assert p.loc["A", "beta_pool"] > b0a and p.loc["B", "beta_pool"] > 0.08 / 0.8
    assert p["shared_scale"].iloc[0] > 1 and p["own_weight"].between(0, 1).all()
    assert p.loc["B", "own_weight"] == 0 and p.loc["A", "own_weight"] > 0
