"""Input validation: bad files fail loudly, naming the problem."""

import numpy as np
import pandas as pd
import pytest

from mortgage_elasticity import panel, runlog
from mortgage_elasticity.config import Config

A, B = "OO_PI_70-80", "INV_IO_>90"
PANEL = pd.DataFrame({
    "segment": [A] * 3 + [B] * 3,
    "period": ["2025-01-05", "2025-01-12", "2025-01-19"] * 2,
    "lodg_mkt": 1000, "lodg_us": [100, 110, 120, 50, 55, 60], "apps": 100,
    "headline_cum_bps": 0.0, "discretionary_cum_bps": 0.0,
})
EVENTS = pd.DataFrame({"date": ["2025-01-09", "2025-01-12"], "segments": ["ALL", f"{B}; {A}"],
                       "type": ["price", "Other"], "description": ["repricing", "campaign"]})


def _write(tmp_path, df, name="f.csv"):
    df.to_csv(tmp_path / name, index=False)
    return tmp_path / name


def test_good_files_load(tmp_path):
    df = panel.load_panel(_write(tmp_path, PANEL))
    ev = panel.load_events(_write(tmp_path, EVENTS, "e.csv"), [A, B])
    assert len(df) == 6
    assert ev["segments"].tolist() == [(A, B), (A, B)] and ev["all"].all()
    assert ev["type"].tolist() == ["price", "other"]
    prep = panel.prepare(df, ev, None, None, Config())
    assert prep.events["t"].tolist() == [1, 1]  # a mid-week date lands in that week
    assert prep.shape == (3, 2) and np.allclose(prep.s0, [(50 + 55 + 60) / 3000, 330 / 3000])


@pytest.mark.parametrize("mutate, match", [
    (lambda d: pd.concat([d, d.iloc[[0]]]), "duplicate"),
    (lambda d: d.assign(lodg_us=d["lodg_us"].where(d.index != 2)), "blank or non-numeric"),
    (lambda d: d.drop(index=5), "balanced"),
    (lambda d: d.assign(period=["05/01/2025", "12/01/2025", "19/01/2025"] * 2), "unparseable"),
    (lambda d: d.assign(segment=["A[1]"] * 3 + [B] * 3), r"\["),
    (lambda d: d.assign(lodg_us=d["lodg_mkt"] + 1), "lodg_us > lodg_mkt"),
    (lambda d: d.assign(apps=-1), "negative"),
    (lambda d: d.assign(lodg_us=10.5), "whole number"),
    (lambda d: d.drop(columns="apps"), "missing columns"),
])
def test_bad_panel_raises(tmp_path, mutate, match):
    with pytest.raises(ValueError, match=match):
        panel.load_panel(_write(tmp_path, mutate(PANEL.copy())))


def test_zero_weeks_are_kept_and_no_market_weeks_leave_the_likelihood(tmp_path):
    d = PANEL.assign(lodg_us=[0, 110, 120, 0, 55, 60], lodg_mkt=[1000] * 3 + [0, 1000, 1000])
    runlog.start()
    df = panel.load_panel(_write(tmp_path, d))
    assert len(df) == 6 and df["share"].isna().sum() == 1
    prep = panel.prepare(df, panel.load_events(_write(tmp_path, EVENTS, "e.csv"), [A, B]), None,
                         None, Config())
    assert prep.obs.sum() == 5
    assert any("1 segment-weeks with zero market lodgements" in line for line in runlog._lines)


def test_discretionary_column_is_optional_and_logged(tmp_path):
    runlog.start()
    df = panel.load_panel(_write(tmp_path, PANEL.drop(columns="discretionary_cum_bps")))
    assert (df["discretionary_cum_bps"] == 0).all()
    assert any("no discretionary_cum_bps column" in line for line in runlog._lines)


def test_date_format_setting(tmp_path):
    d = PANEL.assign(period=["05/01/2025", "12/01/2025", "19/01/2025"] * 2)
    df = panel.load_panel(_write(tmp_path, d), date_format="%d/%m/%Y")
    assert df["period"].min() == pd.Timestamp("2025-01-05")


@pytest.mark.parametrize("mutate, match", [
    (lambda e: e.assign(type=["price", "campaign"]), "type must be price or other"),
    (lambda e: e.assign(segments=["ALL", "Z"]), "not in the panel"),
    (lambda e: e.assign(segments=["ALL", None]), "blank segments"),
    (lambda e: e.drop(columns="type"), "missing columns"),
    (lambda e: pd.concat([e, e.iloc[[0]]]), "duplicate"),
])
def test_bad_events_raise(tmp_path, mutate, match):
    with pytest.raises(ValueError, match=match):
        panel.load_events(_write(tmp_path, mutate(EVENTS.copy())), [A, B])


def test_segment_names_carry_the_pooling_dimensions():
    bands = Config().lvr_bands
    d = panel.segment_dims([A, B, "INV_PI_<=60"], bands)
    assert d.loc[B].tolist() == ["INV", "IO", ">90"] and d.loc["INV_PI_<=60", "lvr_band"] == "<=60"
    for bad in ["OO_PI", "OO_FIX_70-80", "OO_PI_75-80", "OOPI70-80"]:
        with pytest.raises(ValueError, match="PURPOSE_REPAYMENT_LVRBAND"):
            panel.segment_dims([bad], bands)
