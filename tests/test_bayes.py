"""Optional PyMC panel model: samples cleanly and its intervals cover the true betas.
Skipped unless the extra is installed (`uv sync --extra bayes`)."""

import pandas as pd
import pytest

pytest.importorskip("pymc")

from mortgage_elasticity import panel
from mortgage_elasticity.bayes import fit_hierarchical
from mortgage_elasticity.config import Priors
from mortgage_elasticity.simulate import simulate


def test_hierarchical_recovers_betas_on_gap_path(tmp_path):
    pr = Priors()
    truth = simulate(tmp_path, seed=0, competitors=True)
    true = pd.Series({s: b for s, (_, b) in truth["segments"].items()})
    df = panel.load_panel(tmp_path / "panel.csv")
    s0 = panel.baseline_share(df, 13)
    comp = panel.load_competitors(tmp_path / "competitors.csv", (), df["period"].unique())
    df = panel.add_benchmark(df, comp, 1 / true, 8)  # T = 1/beta at the truth: isolate the sampler
    df = panel.add_perceived_price(df, pr.kappa, pr.h_headline, pr.h_discretionary)
    h = fit_hierarchical(df, s0, pr.u, pr.prior_sd_frac, draws=500, tune=500).set_index("segment")
    assert h["r_hat"].max() < 1.02 and h["divergences"].iloc[0] == 0
    t = true[h.index]
    assert ((h["ci90_lo"] <= t) & (t <= h["ci90_hi"])).sum() >= 4  # 90% intervals, 6 segments
    assert ((h["mean"] - t).abs() / t).median() < 0.1
