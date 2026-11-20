"""One small simulated dataset and one fit of it, shared by the model tests (fits are slow)."""

from dataclasses import replace

import pandas as pd
import pytest

from mortgage_elasticity import bayes, panel
from mortgage_elasticity.config import Config
from mortgage_elasticity.simulate import simulate

# small and quick; the full-size runs are scripts/acceptance.py
SMALL = replace(Config(), chains=2, tune=400, draws=400, T_tune=250, T_draws=250, prior_draws=500,
                val_chains=2, val_tune=300, val_draws=300, placebo_runs=2, ppc_draws=100,
                quiet_first_origin=40, quiet_step=20, sens_prior_medians=(0.08,),
                sens_T_factors=(), sens_kappas=())
SIM = {"seed": 1, "periods": 72, "n_segments": 10, "n_bank_moves": 2, "n_segment_moves": 4,
       "n_rba": 1, "n_other": 2, "volume_scale": 3.0}


def load(path, cfg) -> panel.Prepared:
    df = panel.load_panel(path / "panel.csv")
    segs = list(dict.fromkeys(df["segment"]))
    comp = panel.load_competitors(path / "competitors.csv", (), df["period"].unique())
    s0 = df.groupby("segment", sort=False)[["lodg_us", "lodg_mkt"]].sum()
    T = bayes.prior_T((s0["lodg_us"] / s0["lodg_mkt"]).to_numpy(), segs, cfg)
    return panel.prepare(df, panel.load_events(path / "events.csv", segs), comp, T, cfg)


@pytest.fixture(scope="session")
def sim(tmp_path_factory):
    path = tmp_path_factory.mktemp("sim")
    truth = simulate(path, **SIM)
    return path, truth


@pytest.fixture(scope="session")
def fit(sim):
    path, _ = sim
    with pytest.warns() as _:  # benchmark data warnings; short chains
        return bayes.estimate(load(path, SMALL), SMALL)


@pytest.fixture(scope="session")
def truth_u(sim):
    return pd.Series({s: v["u"] for s, v in sim[1]["segments"].items()})
