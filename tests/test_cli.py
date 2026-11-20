"""CLI smoke test: simulate -> check-moves -> estimate -> forecast -> export-excel."""

import sys

import pandas as pd
import pytest
from conftest import SIM, SMALL

from mortgage_elasticity import cli
from mortgage_elasticity.simulate import simulate

pytestmark = pytest.mark.filterwarnings("ignore")


def _run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["elasticity", *map(str, argv)])
    cli.main()


def test_commands_write_their_outputs(tmp_path, monkeypatch, capsys):
    data, out = tmp_path / "data", tmp_path / "out"
    simulate(data, **{**SIM, "n_segments": 6, "periods": 60})
    monkeypatch.setattr(cli, "Config", lambda: SMALL)
    inputs = ["--panel", data / "panel.csv", "--events", data / "events.csv",
              "--competitors", data / "competitors.csv", "--out", out]
    _run(monkeypatch, "check-moves", *inputs)
    moves = pd.read_csv(out / "check_moves.csv")
    assert {"implied_ratio_u=8%/bp", "implied_ratio_u=3%/bp", "apps_ratio"} <= set(moves.columns)

    _run(monkeypatch, "estimate", *inputs)
    for name in ["segments.csv", "global.csv", "natural_share.csv", "accuracy.csv", "fit.pkl",
                 "run_log.txt", "T_path.csv", "diagnostics_estimators.csv", "diagnostics_ppc.csv"]:
        assert (out / name).exists(), name
    seg = pd.read_csv(out / "segments.csv")
    assert {"share_now_p50", "u_p10", "u_p50", "u_p90", "beta_p50", "T_bps", "n_price_moves",
            "prior_share", "conversion", "conv_trend_per_period_pct", "thin_volume",
            "sampler_warning"} <= set(seg.columns)
    assert len(list((out / "natural_share").glob("*.png"))) == len(seg) == 6
    log = (out / "run_log.txt").read_text()
    for needle in ["Config: prior_u_median=0.03", "T loop 1", "full fit:", "Prior check:",
                   "Soft-min temperature by iteration"]:
        assert needle in log, needle

    capsys.readouterr()
    _run(monkeypatch, "forecast", "--fit", out / "fit.pkl", "--segments",
         f"{seg['segment'][0]};{seg['segment'][1]}", "--headline", -10, "--out", out)
    printed = capsys.readouterr().out
    assert "PRICE-EFFECT interval" in printed and "FULL PREDICTIVE interval" in printed
    assert "Long-run effect" in printed
    f = pd.read_csv(out / "forecast.csv")
    assert set(f["segment"]) == {seg["segment"][0], seg["segment"][1], "TOTAL"}
    assert pd.read_csv(out / "forecast_weekly.csv")["week"].max() == SMALL.horizon

    _run(monkeypatch, "export-excel", "--fit", out / "fit.pkl", "--out", out / "e.xlsx")
    assert (out / "e.xlsx").exists()
