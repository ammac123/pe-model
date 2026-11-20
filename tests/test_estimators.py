"""Event study, fixed effects and grid on small deterministic panels."""

import numpy as np
import pandas as pd
import pytest

from mortgage_elasticity import events, fe, panel
from mortgage_elasticity.simulate import simulate

PERIODS = pd.date_range("2025-01-05", periods=30, freq="W-SUN")


def _panel(x_by_seg: dict, beta: float = 0.05) -> pd.DataFrame:
    rows = []
    for k, (seg, x) in enumerate(x_by_seg.items()):
        rows.append(pd.DataFrame({"segment": seg, "period": PERIODS, "x": x,
                                  "y": -2.0 + 0.1 * k - beta * np.asarray(x)}))
    return pd.concat(rows, ignore_index=True)


def _step(t: int, size: float) -> np.ndarray:
    return np.where(np.arange(len(PERIODS)) >= t, size, 0.0)


def _events(segs, when) -> pd.DataFrame:
    return pd.DataFrame({"segment": segs, "period": when, "reason": "funding", "clean": True})


def test_event_study_exact_on_noiseless_panel_and_snaps_dates():
    # controls also moved (an unlogged market move): exact when betas are equal
    df = _panel({"A": _step(15, -10.0), "B": _step(15, -4.0), "C": _step(15, -4.0)})
    day = pd.Timedelta(days=1)
    ev = _events(["A", "A"], [PERIODS[15] - 3 * day, PERIODS[-1] + day])
    es, summary = events.event_study(df, ev, pre=4, post=8,
                                     beta_sd=pd.Series(0.01, index=["A", "B", "C"]))
    assert len(es) == 1 and np.isclose(es["beta_e"].iloc[0], 0.05, rtol=1e-12)
    assert es["period"].iloc[0] == PERIODS[15]  # mid-week date snaps to the period end
    assert es["beta_e_se"].iloc[0] > 0
    assert "after_end=1" in summary


def test_event_study_drops_events_without_controls():
    df = _panel({"A": _step(15, -10.0), "B": _step(15, -5.0)})
    es, summary = events.event_study(df, _events(["A", "B"], [PERIODS[15]] * 2), pre=4, post=8)
    assert len(es) == 0 and "no_controls=2" in summary


def test_other_changes_and_veto_make_an_event_unusable():
    df = _panel({"A": _step(10, -10.0), "B": 0.0, "C": 0.0})
    df["y"] += np.random.default_rng(0).normal(0, 0.01, len(df))  # noise, so se > 0
    ev = _events(["A"] * 3, [PERIODS[10]] * 3).assign(other_changes=["", "campaign", ""],
                                                      clean=[True, True, False])
    es, _ = events.event_study(df, ev, pre=4, post=8)  # same move logged three ways
    assert es["clean"].tolist() == [True, False, False]
    assert es["flagged_clean"].tolist() == [True, True, False]
    assert es["other_changes"].tolist() == ["", "campaign", ""]


def test_twfe_rank_check_and_flat_segment():
    x = _step(10, -10.0) + _step(20, 5.0)
    with pytest.raises(ValueError, match="cross-segment variation"):
        fe.fit_twfe(_panel({"A": x, "B": x, "C": x}))
    rng = np.random.default_rng(0)
    df = _panel({"A": x, "B": _step(14, 10.0), "C": np.zeros(len(PERIODS))})
    df["y"] += rng.normal(0, 0.01, len(df))
    b = fe.betas_from_fit(fe.fit_twfe(df)).set_index("segment")
    assert np.isnan(b.loc["C", "beta_fe"]) and np.isnan(b.loc["C", "beta_fe_se"])
    assert np.isclose(b.loc["A", "beta_fe"], 0.05, atol=0.01)
    # move model: a segment that never repriced gets a borrowed x after demeaning -> still NaN
    raw = df.assign(headline_cum_bps=df["x"], discretionary_cum_bps=0.0)
    dm = panel.add_perceived_price(raw, 0.5, 1.0, 1.0, demean=True)
    assert np.isnan(fe.betas_from_fit(fe.fit_twfe(dm)).set_index("segment").loc["C", "beta_fe"])


def test_grid_keeps_prior_unless_gain_is_large(tmp_path):
    simulate(tmp_path, seed=3)
    df = panel.load_panel(tmp_path / "panel.csv")
    prior = {"kappa": 0.5, "hH": 2.0, "hD": 8.0}
    grid, best, _ = fe.grid_search(df, (0.3, 0.7), (1.0, 4.0), (8.0,), prior, min_gain=1e9)
    assert best == prior and len(grid) == 4
    _, best, _ = fe.grid_search(df, (0.3, 0.7), (1.0, 4.0), (8.0,), prior, min_gain=-1.0)
    assert best == grid.iloc[0][["kappa", "hH", "hD"]].to_dict()
