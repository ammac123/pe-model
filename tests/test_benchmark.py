"""Competitor benchmark: soft-min, loading competitor files, perceived price."""

import numpy as np
import pandas as pd
import pytest

from mortgage_elasticity import panel
from mortgage_elasticity.model import lag, softmin_rate

PERIODS = pd.date_range("2025-01-05", periods=4, freq="W-SUN")


def test_softmin():
    T = 10.0
    assert np.isclose(softmin_rate([[600.0]], T)[0], 600)
    assert np.isclose(softmin_rate([[600.0, 600.0]], T)[0], 600 - T * np.log(2))
    a = np.array([[600.0, 610.0, 625.0]])
    assert np.isclose(softmin_rate(a + 50, T)[0], softmin_rate(a, T)[0] + 50)
    assert softmin_rate(np.c_[a, [[700.0]]], T)[0] < softmin_rate(a, T)[0]  # never raises r*
    assert np.isclose(softmin_rate([[600.0, np.nan]], T)[0], 600)
    assert softmin_rate(a, 0)[0] == 600
    assert np.isclose(softmin_rate(a, 1e-6)[0], 600)  # no overflow for tiny T
    assert np.isnan(softmin_rate([[np.nan, np.nan]], T)[0])
    # bounds: between the cheapest rate less T log N and the cheapest rate
    assert a.min() - T * np.log(a.size) <= softmin_rate(a, T)[0] <= a.min()
    assert np.isclose(softmin_rate([[600.0, 1e6]], T)[0], 600)  # stable for a far-off rate


def test_gap_and_perceived_price():
    df = pd.DataFrame([{"segment": s, "period": p, "headline_cum_bps": h,
                        "discretionary_cum_bps": d}
                       for s, hs in {"A": [0, -10, -10, -10], "B": [0, 0, 5, 5]}.items()
                       for p, h, d in zip(PERIODS, hs, [0, 0, -10, -10])])
    a = df[df.segment == "A"]
    own = a["headline_cum_bps"] + 0.5 * a["discretionary_cum_bps"]
    assert np.allclose(panel.add_gap(df, 0.5).loc[a.index, "gap_raw"], own)
    # with a benchmark: its move since the first period is subtracted
    g = panel.add_gap(df.assign(r_star_bps=600.0 - 10 * (df.period > PERIODS[1])), 0.5)
    assert np.allclose(g.loc[a.index, "gap_raw"], own - np.array([0, 0, -10, -10]))
    x = panel.add_perceived_price(g, 2.0)  # one half-life for every component
    assert np.allclose(x.loc[a.index, "x"], lag(g.loc[a.index, "gap_raw"], 2.0))


COMP = pd.DataFrame({
    "period": [str(p.date()) for p in PERIODS for _ in range(3)],
    "segment": "A", "lender": ["L1", "L2", "L3"] * 4, "rate_pct": 6.2, "adj_bps": 5.0,
})


def _write(tmp_path, df, name="c.csv"):
    df.to_csv(tmp_path / name, index=False)
    return tmp_path / name


@pytest.mark.parametrize("mutate, match", [
    (lambda c: c.drop(columns="lender"), "missing"),
    (lambda c: pd.concat([c, c.iloc[[0]]]), "duplicate"),
    (lambda c: c.assign(rate_pct=620.0), "looks like bps"),
])
def test_bad_competitors_raise(tmp_path, mutate, match):
    with pytest.raises(ValueError, match=match):
        panel.load_competitors(_write(tmp_path, mutate(COMP.copy())))


def test_competitors_eligible_exclude_and_ffill(tmp_path):
    c = COMP.assign(eligible=["yes", "no", "yes"] * 4)
    comp = panel.load_competitors(_write(tmp_path, c), exclude=("L3",))
    live = comp[np.isfinite(comp["eff_bps"])]
    assert set(live["lender"]) == {"L1"} and np.isclose(live["eff_bps"].iloc[0], 615)
    df = pd.DataFrame({"segment": "A", "period": PERIODS})
    T = pd.Series({"A": 10.0})
    sparse = comp[comp["period"] == PERIODS[0]]  # only the first period reported
    with pytest.warns(UserWarning):
        out = panel.add_benchmark(df, sparse, T, ffill=3)
    assert np.allclose(out["r_star_bps"], 615)
    with pytest.raises(ValueError, match="no lender rates"):
        panel.add_benchmark(df, sparse, T, ffill=1)
    with pytest.raises(ValueError, match="no rows for panel segment"):
        panel.add_benchmark(df.assign(segment="B"), comp, pd.Series({"B": 10.0}), ffill=3)


def test_ineligible_rate_is_not_carried_forward(tmp_path):
    # L2 is cheapest while eligible, then withdrawn: r* must jump to L1, not keep L2's old rate
    c = pd.DataFrame({"period": list(PERIODS[:2]) * 2, "segment": "A",
                      "lender": ["L1", "L1", "L2", "L2"], "rate_pct": [6.2, 6.2, 6.0, 6.0],
                      "eligible": ["yes", "yes", "yes", "no"]})
    comp = panel.load_competitors(_write(tmp_path, c))
    df = pd.DataFrame({"segment": "A", "period": PERIODS[:2]})
    with pytest.warns(UserWarning):  # lender count changes 2 -> 1
        out = panel.add_benchmark(df, comp, pd.Series({"A": 0.0}), ffill=8)
    assert out["r_star_bps"].tolist() == [600.0, 620.0]
    log = c.rename(columns={"period": "effective_date"})
    comp = panel.load_competitors(_write(tmp_path, log), periods=PERIODS)
    with pytest.warns(UserWarning):
        out = panel.add_benchmark(pd.DataFrame({"segment": "A", "period": PERIODS}), comp,
                                  pd.Series({"A": 0.0}), ffill=8)
    # withdrawn on the last day of period 2: still eligible for 6 of its 7 days
    assert out["r_star_bps"].tolist() == [600.0, 600.0, 620.0, 620.0]


def test_change_log_is_day_weighted(tmp_path):
    # weekly periods ending Sundays; L1 moves from 6.00% to 6.25% on Thursday of week 2:
    # (Jan 6..Jan 12] -> 3 days at 6.00 (Mon-Wed) + 4 days at 6.25 (Thu-Sun)
    log = pd.DataFrame({"effective_date": ["2024-12-01", "2025-01-09"], "segment": "A",
                        "lender": "L1", "rate_pct": [6.00, 6.25]})
    comp = panel.load_competitors(_write(tmp_path, log), periods=PERIODS)
    rate = comp.set_index("period")["rate_pct"]
    assert np.isclose(rate[PERIODS[0]], 6.00) and np.isclose(rate[PERIODS[2]], 6.25)
    assert np.isclose(rate[PERIODS[1]], (3 * 6.00 + 4 * 6.25) / 7)
    # before a lender's first change its rate is unknown
    late = log.assign(effective_date=["2025-01-09", "2025-01-16"])
    comp = panel.load_competitors(_write(tmp_path, late), periods=PERIODS)
    assert comp["period"].min() == PERIODS[1]
    assert np.isclose(comp.set_index("period")["rate_pct"][PERIODS[1]], 6.00)
