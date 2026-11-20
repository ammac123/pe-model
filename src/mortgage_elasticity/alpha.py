"""Step 6 (v2.5): back out the natural position alpha_kt with beta held fixed, then smooth it.

    alpha_hat_kt = y_kt + beta_k * x_kt      (inversion)
    alpha_hat = alpha + eps,  alpha_t = alpha_{t-1} + eta   (local level)

Do not re-estimate beta here: a free beta and a free drifting alpha trade off against each other.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents


def _filter_fixed_ratio(a: np.ndarray, q_over_r: float) -> np.ndarray:
    r = np.var(np.diff(a)) / 2 if len(a) > 2 else 1.0
    q = q_over_r * r
    out, m, p = np.empty_like(a), a[0], r
    for i, y in enumerate(a):
        if i:
            p += q
            k = p / (p + r)
            m, p = m + k * (y - m), (1 - k) * p
        out[i] = m
    return out


def track_alpha(df: pd.DataFrame, betas: pd.Series, min_periods_mle: int,
                q_over_r: float) -> pd.DataFrame:
    parts = []
    for seg, g in df.groupby("segment", sort=False):
        if not np.isfinite(betas.get(seg, np.nan)):
            print(f"natural_share: skipping {seg} (no finite beta)")
            continue
        a_hat = (g["y"] + betas[seg] * g["x"]).to_numpy()
        if len(a_hat) >= min_periods_mle:
            res = UnobservedComponents(a_hat, level="local level").fit(disp=False)
            level, method = res.smoothed_state[0], "mle_smoothed"
        else:
            level, method = _filter_fixed_ratio(a_hat, q_over_r), "fixed_q_over_r_filtered"
        parts.append(pd.DataFrame({
            "segment": seg, "period": g["period"].to_numpy(), "share_obs": g["share"].to_numpy(),
            "alpha_hat": a_hat, "alpha_level": level,
            "natural_share": 1 / (1 + np.exp(-level)), "method": method,
        }))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
