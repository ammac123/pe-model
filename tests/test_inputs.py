"""Input validation: bad files fail loudly, naming the problem."""

import pandas as pd
import pytest

from mortgage_elasticity import panel

PANEL = pd.DataFrame({
    "segment": ["A", "A", "A", "B", "B", "B"],
    "period": ["2025-01-05", "2025-01-12", "2025-01-19"] * 2,
    "lodg_mkt": 1000, "lodg_us": [100, 110, 120, 50, 55, 60], "apps": 100,
    "headline_cum_bps": 0.0, "discretionary_cum_bps": 0.0,
})
EVENTS = pd.DataFrame({"segment": ["A"], "period": ["2025-01-12"], "reason": ["funding"],
                       "clean": ["TRUE"]})


def _write(tmp_path, df, name="f.csv"):
    df.to_csv(tmp_path / name, index=False)
    return tmp_path / name


def test_good_files_load(tmp_path):
    df = panel.load_panel(_write(tmp_path, PANEL))
    ev = panel.load_events(_write(tmp_path, EVENTS, "e.csv"), df["segment"].unique())
    assert len(df) == 6 and ev["clean"].tolist() == [True]


@pytest.mark.parametrize("mutate, match", [
    (lambda d: pd.concat([d, d.iloc[[0]]]), "duplicate"),
    (lambda d: d.assign(lodg_us=d["lodg_us"].where(d.index != 2)), "blank or non-numeric"),
    (lambda d: d.drop(index=5), "balanced"),
    (lambda d: d.assign(period=["05/01/2025", "12/01/2025", "19/01/2025"] * 2), "unparseable"),
    (lambda d: d.assign(segment=["A[1]"] * 3 + ["B"] * 3), r"\["),
    (lambda d: d.assign(lodg_us=d["lodg_mkt"]), "outside"),
])
def test_bad_panel_raises(tmp_path, mutate, match):
    with pytest.raises(ValueError, match=match):
        panel.load_panel(_write(tmp_path, mutate(PANEL.copy())))


def test_date_format_setting(tmp_path):
    d = PANEL.assign(period=["05/01/2025", "12/01/2025", "19/01/2025"] * 2)
    df = panel.load_panel(_write(tmp_path, d), date_format="%d/%m/%Y")
    assert df["period"].min() == pd.Timestamp("2025-01-05")


@pytest.mark.parametrize("mutate, match", [
    (lambda e: e.assign(clean=["maybe"]), "unrecognised clean"),
    (lambda e: e.assign(segment=["Z"]), "not in the panel"),
])
def test_bad_events_raise(tmp_path, mutate, match):
    with pytest.raises(ValueError, match=match):
        panel.load_events(_write(tmp_path, mutate(EVENTS.copy())), ["A", "B"])


def test_clean_parser_is_strict_but_flexible(tmp_path):
    e = pd.concat([EVENTS] * 4).assign(clean=[" Yes", "0", "1.0", "f"],
                                       period=PANEL["period"].tolist()[:3] + ["2025-01-26"])
    assert panel.load_events(_write(tmp_path, e))["clean"].tolist() == [True, False, True, False]


def test_clean_and_other_changes_are_optional(tmp_path):
    ev = panel.load_events(_write(tmp_path, EVENTS.drop(columns="clean")))
    assert ev["clean"].tolist() == [True] and ev["other_changes"].tolist() == [""]
    e = pd.concat([EVENTS] * 4).assign(other_changes=[" Campaign ", None, "None", "n/a"],
                                       period=PANEL["period"].tolist()[:3] + ["2025-01-26"])
    assert panel.load_events(_write(tmp_path, e))["other_changes"].tolist() == ["Campaign", "", "", ""]
