"""End to end on synthetic data: the pipeline should recover the true betas roughly."""

import numpy as np
import pandas as pd

from mortgage_elasticity import events, fe, panel, pool
from mortgage_elasticity.simulate import simulate


def test_recovers_betas(tmp_path):
    truth = simulate(tmp_path, seed=3)
    df = panel.load_panel(tmp_path / "panel.csv")
    ev = panel.load_events(tmp_path / "events.csv")
    dfx = panel.add_perceived_price(df, truth["kappa"], truth["hH"], truth["hD"])

    fe_b = fe.betas_from_fit(fe.fit_twfe(dfx)).set_index("segment")["beta_fe"]
    true_b = {k: v[1] for k, v in truth["segments"].items()}
    err = np.array([fe_b[k] / true_b[k] - 1 for k in true_b])
    assert np.median(np.abs(err)) < 0.25

    es, _ = events.event_study(dfx, ev, pre=4, post=8)
    assert len(es) > 0
    assert np.median(es["beta_e"]) > 0


def _fe_err(dfx, true_b):
    b = fe.betas_from_fit(fe.fit_twfe(dfx)).set_index("segment")["beta_fe"]
    return np.median(np.abs(b / true_b[b.index] - 1))


def test_gap_model_recovers_betas(tmp_path):
    truth = simulate(tmp_path, seed=0, competitors=True)
    true_b = pd.Series({k: v[1] for k, v in truth["segments"].items()})
    df = panel.load_panel(tmp_path / "panel.csv")
    comp = panel.load_competitors(tmp_path / "competitors.csv")
    k, hH, hD = truth["kappa"], truth["hH"], truth["hD"]
    s0 = panel.baseline_share(df, 13)
    gap = {name: _fe_err(panel.add_perceived_price(
        panel.add_benchmark(df, comp, T, 8), k, hH, hD), true_b)
        for name, T in [("oracle", 1 / true_b), ("prior", (1 - s0) / 0.08)]}
    move = _fe_err(panel.add_perceived_price(df, k, hH, hD), true_b)  # pre-change behaviour
    assert gap["oracle"] < 0.15
    assert gap["oracle"] < move and gap["prior"] < move


def test_reactive_repricing_biases_fe_and_unvetoed_events(tmp_path):
    """Cuts made because volume fell short bias beta_fe towards zero. Events are only unbiased if
    those cuts are vetoed (clean=FALSE); logged as usable they pull beta_post down too."""
    fe_err, veto_err, all_err = [], [], []
    for seed in range(10):
        truth = simulate(tmp_path, seed=seed, reactive=True)
        true_b = pd.Series({k: v[1] for k, v in truth["segments"].items()})
        df = panel.load_panel(tmp_path / "panel.csv")
        dfx = panel.add_perceived_price(df, truth["kappa"], truth["hH"], truth["hD"], demean=True)
        b = fe.betas_from_fit(fe.fit_twfe(dfx)).set_index("segment")["beta_fe"]
        fe_err.append(np.median(b / true_b[b.index] - 1))
        ev = panel.load_events(tmp_path / "events.csv")
        for ev_, errs in [(ev, veto_err), (ev.assign(clean=True), all_err)]:
            es, _ = events.event_study(dfx, ev_, 4, 8)
            post = pool.posterior_betas(panel.baseline_share(df, 13), es, 0.08, prior_sd_frac=10.0)
            errs.append(np.median(post.set_index("segment")["beta_post"] / true_b - 1))
    fe_bias, veto_bias, all_bias = np.mean(fe_err), np.mean(veto_err), np.mean(all_err)
    assert fe_bias < 0
    assert abs(veto_bias) < 0.5 * abs(fe_bias)
    assert all_bias < -0.1  # README: about -16 to -18% when every move is logged as usable
