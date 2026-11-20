"""The model end to end on a small simulated panel: T loop, recovery, outputs, forecast."""

import numpy as np
import pandas as pd
import pytest
from conftest import SMALL

from mortgage_elasticity import diagnostics, report, validate
from mortgage_elasticity.forecast import accuracy_table, forecast, paths
from mortgage_elasticity.model import multiplier

pytestmark = pytest.mark.filterwarnings("ignore")


def test_T_loop_converges_to_one_over_beta(fit, sim):
    assert fit.T_converged and fit.T_path.shape[1] >= 3  # segment, start, at least one iteration
    beta = np.median(fit.draws["beta"], axis=0)
    assert np.allclose(fit.prep.T.to_numpy() * beta, 1, atol=0.25)  # T = 1 / beta at the end
    true_T = pd.Series({s: 1 / v["beta"] for s, v in sim[1]["segments"].items()})
    assert ((fit.prep.T / true_T[fit.prep.segs] - 1).abs() < 0.35).all()


def test_recovers_the_true_sensitivity(fit, truth_u, sim):
    lo, mid, hi = fit.q("u")
    t = truth_u[fit.prep.segs].to_numpy()
    assert np.median(np.abs(mid / t - 1)) < 0.15
    assert ((lo <= t) & (t <= hi)).mean() >= 0.5  # 80% intervals; small panel, short chains
    assert abs(np.exp(np.median(fit.draws["mu_u"])) / sim[1]["u"] - 1) < 0.25
    assert abs(np.median(fit.draws["h"]) - 2.0) < 0.7


def test_fitted_lag_matches_the_recursion(fit):
    from mortgage_elasticity.model import lag
    x, k = fit.x(), 3
    assert np.allclose(x[7, :, k], lag(fit.prep.gap_raw[:, k], fit.draws["h"][7]), atol=1e-3)


def test_estimate_outputs(fit):
    seg = report.segments_table(fit)
    assert len(seg) == len(fit.prep.segs)
    assert (seg["u_p10"] < seg["u_p50"]).all() and (seg["u_p50"] < seg["u_p90"]).all()
    assert seg["prior_share"].between(0, 1).all()
    assert (seg["prior_share"] < 0.5).all()  # informative data: the answer is not the prior
    assert np.allclose(seg["beta_p50"], seg["u_p50"] / (1 - fit.prep.s0))
    assert seg["thin_volume"].any() and not seg["thin_volume"].all()
    g = report.global_table(fit).set_index("parameter")
    assert {"h", "mu_u_pct_per_bp", "a_inv", "a_io", "sigma_g", "sigma_v", "c",
            "c_lvr[>90]"} <= set(g.index)
    nat = report.natural_share_table(fit)
    assert len(nat) == fit.prep.M.size
    assert (nat["natural_share_p10"] <= nat["natural_share_p90"]).all()


def test_forecast_paths_and_intervals(fit):
    seg = fit.prep.segs[0]
    cut = forecast(fit, seg, -10).summary.set_index("version")
    pe, full = cut.loc["price_effect"], cut.loc["full_predictive"]
    assert 1 < pe["multiplier_p10"] < pe["multiplier_p50"] < pe["multiplier_p90"]
    # a move is still phasing in over 12 weeks: less than the long-run effect
    assert pe["multiplier_p50"] < pe["long_run_multiplier_p50"]
    # drift and count noise only ever widen the interval
    assert (full["apps_p90"] - full["apps_p10"]) > (pe["apps_p90"] - pe["apps_p10"])
    assert np.isclose(pe["dx_bps"], -10)
    rise = forecast(fit, seg, 10).summary.set_index("version").loc["price_effect"]
    assert rise["multiplier_p90"] < 1
    none = forecast(fit, seg, 0, settled=True).summary.set_index("version").loc["price_effect"]
    assert np.isclose(none["multiplier_p10"], 1) and np.isclose(none["multiplier_p90"], 1)
    # discretionary changes count at kappa; a cheaper competitor raises the gap
    half = forecast(fit, seg, 0, d_discretionary_bps=-20).summary.iloc[0]
    assert np.isclose(half["dx_bps"], -10)
    lender = fit.prep.comp.loc[fit.prep.comp["segment"] == seg, "lender"].iloc[0]
    assert forecast(fit, seg, 0, competitor_changes={lender: -25}).summary.iloc[0]["dx_bps"] > 0
    with pytest.raises(ValueError, match="not in the benchmark"):
        forecast(fit, seg, 0, competitor_changes={"NOBODY": -25})


def test_forecast_matches_the_multiplier_formula_and_sums_groups(fit):
    p = paths(fit, -10, settled=True)
    d, k = fit.draws, 2
    # settled baseline: weekly multiplier = 1 / (s + (1 - s) exp(beta dx (1 - (1 - phi)^tau)))
    felt = 1 - 2.0 ** (-np.arange(1, 13) / d["h"][:, None])
    want = multiplier(-10 * felt, p.share_base[:, :, k], d["beta"][:, [k]])
    assert np.allclose(p.share_scn[:, :, k] / p.share_base[:, :, k], want)
    f = forecast(fit, fit.prep.segs[:3], -10).summary
    tot = f[(f["segment"] == "TOTAL") & (f["version"] == "price_effect")].iloc[0]
    parts = f[(f["segment"] != "TOTAL") & (f["version"] == "price_effect")]
    assert parts["apps_p50"].sum() == pytest.approx(tot["apps_p50"], rel=0.02)
    assert parts["multiplier_p50"].min() < tot["multiplier_p50"] < parts["multiplier_p50"].max()


def test_accuracy_table(fit):
    acc = accuracy_table(fit)
    n_units = len(fit.prep.segs) + 1 + sum(fit.prep.dims[c].nunique()
                                           for c in ("purpose", "repayment", "lvr_band"))
    assert len(acc) == n_units * len(SMALL.accuracy_moves_bps)
    one = acc[(acc["name"] == fit.prep.segs[0]) & (acc["kind"] == "segment")].set_index("move_bps")
    pe, full = "apps_halfwidth_rel_price_effect", "apps_halfwidth_rel_full_predictive"
    assert one.loc[-25, pe] > one.loc[-5, pe]  # bigger moves are less certain
    assert (acc[full] > acc[pe]).all()
    assert (acc["meets_5pct_price_effect"] == (acc[pe] <= 0.05)).all()
    everything = acc[acc["kind"] == "all"].set_index("move_bps")
    assert everything.loc[-5, full] < one.loc[-5, full]  # groups are steadier than one segment


def test_diagnostics(fit, tmp_path, truth_u):
    moves = diagnostics.raw_move_check(fit.prep, SMALL, tmp_path)
    n_moves = moves["move"].nunique()
    assert n_moves >= 5 and (moves["segment"] == "TOTAL").sum() == n_moves
    assert len(list(tmp_path.glob("move_*.png"))) == n_moves == len(list(tmp_path.glob("*.csv")))
    cut = moves[(moves["segment"] == "TOTAL") & (moves["d_gap_bps"] < -4)]
    assert (cut["implied_ratio_u=8%/bp"] > cut["implied_ratio_u=3%/bp"]).all()
    est = diagnostics.estimators(fit)
    assert est["u_twfe"].notna().any() and "towards zero" in est["twfe_known_bias"].iloc[0]
    ppc = diagnostics.ppc(fit)
    assert set(ppc["statistic"]) == {"share_variance", "lag1_autocorrelation"}
    assert ppc["inside"].mean() > 0.5


def test_validation_and_sensitivity_run(fit):
    ev = validate.events(fit)
    tot = ev[ev["segment"] == "TOTAL"]
    assert len(tot) == len(fit.prep.moves("price")) and tot["rel_error"].abs().median() < 0.25
    q = validate.quiet(fit)
    assert len(q) and {"origin_week", "covered", "rel_error"} <= set(q.columns)
    pl = validate.placebo(fit)
    assert len(pl) == SMALL.placebo_runs and (pl["coef_low"] < pl["coef_high"]).all()
    s = validate.sensitivity(fit)
    assert list(s["run"].unique()) == ["base", "prior median 8%/bp",
                                       "week dummies (time_effect=fe)", "jumps off"]
    overall = s[s["kind"] == "all"].set_index("run")
    # the data, not the prior, set the answer: an 8%/bp prior moves u by far less than 8/3
    assert abs(overall.loc["prior median 8%/bp", "u_p50_vs_base"]) < 0.25
