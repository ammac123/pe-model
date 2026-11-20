"""Diagnostic only: two-way fixed effects  y_kt = alpha_k + delta_t - beta_k * x_kt + e.
Nothing here feeds the estimate (bayes.py); diagnostics.py reports it beside the model's u_k."""

import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def _flat_segments(d: pd.DataFrame) -> set:
    """Segments whose x never moves: their slope is not identified."""
    rng = d.groupby("segment")["x"].agg(lambda v: v.max() - v.min())
    return set(rng.index[rng < 1e-9])


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
    flat = _flat_segments(res.model.data.frame)
    rows = []
    for name, coef in res.params.items():
        m = re.fullmatch(r"C\(segment\)\[(?:T\.)?(.+)\]:x", name)
        if m:
            seg = m.group(1)
            b, se = (np.nan, np.nan) if seg in flat else (-coef, res.bse[name])
            rows.append({"segment": seg, "beta_fe": b, "beta_fe_se": se})
    return pd.DataFrame(rows)
