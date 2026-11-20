"""Forward model: the formulas the pricing tool (Excel) applies."""

import numpy as np


def beta_from_uplift(u: float, s0: float) -> float:
    """Calibration A: match a linear uplift u (fraction of apps per bp) at the margin."""
    return u / (1.0 - s0)


def u_from_beta(beta: float, s0: float) -> float:
    """Implied marginal uplift at baseline."""
    return beta * (1.0 - s0)


def multiplier(x, s0: float, beta: float):
    """Application multiplier for a cumulative perceived price change x (bps, + = dearer)."""
    x = np.asarray(x, dtype=float)
    return 1.0 / (s0 + (1.0 - s0) * np.exp(beta * x))


def softmin_rate(eff, T: float):
    """Market benchmark r* = -T log sum_j exp(-eff_j / T) per row (periods x lenders, bps, NaN =
    absent). This is the MNL inclusive value with T = 1/beta: adding a lender lowers r*. T <= 0
    gives the hard min. Rows with no lenders come back NaN (callers must check)."""
    eff = np.atleast_2d(np.asarray(eff, dtype=float))
    out = np.full(len(eff), np.nan)
    has = ~np.isnan(eff).all(axis=1)
    if T <= 0:
        out[has] = np.nanmin(eff[has], axis=1)
        return out
    z = np.where(np.isnan(eff[has]), -np.inf, -eff[has] / T)
    m = z.max(axis=1, keepdims=True)  # shift for a stable log-sum-exp
    out[has] = -T * (m[:, 0] + np.log(np.exp(z - m).sum(axis=1)))
    return out


def lag(values, half_life: float):
    """Partial adjustment towards the actual cumulative price; starts at the first value (steady state)."""
    values = np.asarray(values, dtype=float)
    lam = 1.0 - 2.0 ** (-1.0 / half_life)
    out = np.empty_like(values)
    p = values[0]
    for i, v in enumerate(values):
        p = p + lam * (v - p)
        out[i] = p
    return out
