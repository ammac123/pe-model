"""CLI smoke test and the Excel contract: original `parameters` columns stay, in order."""

import argparse

import numpy as np
import openpyxl
import pandas as pd
import pytest

from mortgage_elasticity import cli
from mortgage_elasticity.cli import cmd_estimate
from mortgage_elasticity.simulate import simulate

ORIGINAL = [
    "segment", "s0", "beta_prior", "beta_prior_sd", "n_clean_events", "beta_post", "beta_post_sd",
    "beta_ci90_lo", "beta_ci90_hi", "prior_weight", "beta_fe", "beta_fe_se", "u_post", "u_fe",
    "kappa", "h_headline", "h_discretionary",
]
NEW = [
    "beta_pool", "beta_pool_sd", "beta_pool_ci90_lo", "beta_pool_ci90_hi", "u_pool",
    "shared_scale", "shared_scale_sd", "own_weight", "x_basis", "softmin_T_bps",
    "r_star_now_pct", "n_lenders_now", "x_drift_s0_window",
]


@pytest.mark.parametrize("gap", [False, True])
def test_estimate_writes_the_excel_contract(tmp_path, gap, monkeypatch):
    simulate(tmp_path, seed=0, competitors=True)
    args = argparse.Namespace(
        panel=str(tmp_path / "panel.csv"), events=str(tmp_path / "events.csv"),
        competitors=str(tmp_path / "competitors.csv") if gap else None,
        out=str(tmp_path / "out"), no_grid=True, bayes=False)
    fits = []  # first-pass beta_fe: T is re-solved from it, so the final beta_fe differs a bit
    fit = cli._fit
    monkeypatch.setattr(cli, "_fit", lambda *a: fits.append(fit(*a)) or fits[-1])
    cmd_estimate(args)
    params = pd.read_excel(tmp_path / "out" / "parameters.xlsx", sheet_name="parameters")
    assert list(params.columns) == ORIGINAL + NEW
    assert (params["x_basis"] == ("gap" if gap else "move")).all()
    assert params["r_star_now_pct"].notna().all() == gap
    if gap:  # T re-solved at 1/(first-pass beta_fe), not left at 1/beta_prior
        assert len(fits) == 2
        assert not np.allclose(params["softmin_T_bps"], 1 / params["beta_prior"])
        b1 = fits[0][3].set_index("segment")["beta_fe"].reindex(params["segment"]).to_numpy()
        pos = b1 > 0
        assert pos.any()
        assert np.allclose(params["softmin_T_bps"][pos], 1 / b1[pos], rtol=1e-6)
    sheets = openpyxl.load_workbook(tmp_path / "out" / "parameters.xlsx").sheetnames
    assert {"parameters", "event_estimates", "conversion_check", "natural_share",
            "meta"} <= set(sheets)
    assert ("benchmark" in sheets) == gap
