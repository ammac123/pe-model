"""Step 4a: two-way fixed effects  y_kt = alpha_k + delta_t - beta_k * x_kt + e,
with kappa and the half-lives profiled over a grid."""

import itertools
import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from mortgage_elasticity.panel import add_perceived_price


def _flat_segments(d: pd.DataFrame, own: bool = False) -> set:
    """Segments whose x never moves: their slope is not identified. own=True also flags segments
    whose own price never moved in the move model (demeaning gives them a borrowed x)."""
    cols = ["x"]
    if own and "r_star_bps" not in d and "headline_cum_bps" in d:
        cols = ["headline_cum_bps", "discretionary_cum_bps"]
    rng = d.groupby("segment")[cols].agg(lambda v: v.max() - v.min()).max(axis=1)
    flat = set(rng.index[rng < 1e-9])
    return flat | _flat_segments(d) if own else flat


def fit_twfe(df: pd.DataFrame):
    d = df.copy()
    d["period_id"] = d["period"].astype(str)
    model = smf.ols("y ~ C(segment) + C(period_id) + C(segment):x", data=d)
    # HC1 rather than clustering: with only a handful of segments, clustered SEs are unreliable.
    res = model.fit(cov_type="HC1")
    n_flat = len(_flat_segments(d))  # a quiet segment costs one rank; that's not an error
    if n_flat == d["segment"].nunique() or res.model.rank < model.exog.shape[1] - n_flat:
        raise ValueError("x has no cross-segment variation after period effects; beta_k can't "
                         "be separated from them (identical repricing in every segment?)")
    return res


def betas_from_fit(res) -> pd.DataFrame:
    flat = _flat_segments(res.model.data.frame, own=True)
    rows = []
    for name, coef in res.params.items():
        m = re.fullmatch(r"C\(segment\)\[(?:T\.)?(.+)\]:x", name)
        if m:
            seg = m.group(1)
            b, se = (np.nan, np.nan) if seg in flat else (-coef, res.bse[name])
            rows.append({"segment": seg, "beta_fe": b, "beta_fe_se": se})
    return pd.DataFrame(rows)


def grid_search(df: pd.DataFrame, kappas, hHs, hDs, prior: dict, min_gain: float = 0.0,
                **xkw) -> tuple[pd.DataFrame, dict, float]:
    """Returns the full grid, the combination to use and the SSR gain of the grid winner over the
    priors. A flat SSR means the data can't pin the timing, so the priors are kept unless the
    winner cuts SSR by at least `min_gain`. xkw go to add_perceived_price."""
    rows = []
    for k, hH, hD in itertools.product(kappas, hHs, hDs):
        res = fit_twfe(add_perceived_price(df, k, hH, hD, **xkw))
        rows.append({"kappa": k, "hH": hH, "hD": hD, "ssr": res.ssr})
    grid = pd.DataFrame(rows).sort_values("ssr").reset_index(drop=True)
    grid["ssr_vs_best_pct"] = grid["ssr"] / grid["ssr"].iloc[0] - 1
    ssr_prior = fit_twfe(add_perceived_price(
        df, prior["kappa"], prior["hH"], prior["hD"], **xkw)).ssr
    gain = ssr_prior / grid["ssr"].iloc[0] - 1
    best = grid.iloc[0][["kappa", "hH", "hD"]].to_dict() if gain >= min_gain else dict(prior)
    return grid, best, gain
