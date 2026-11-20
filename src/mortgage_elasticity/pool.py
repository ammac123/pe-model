"""Step 5a: update the judgment prior on beta with usable event estimates (normal-normal).

Prior per segment: beta ~ N(u / (1 - s0_k), (prior_sd_frac * prior mean)^2).
Evidence: usable event estimates beta_e with their standard errors. "Usable" is the `clean` column
written by events.event_study: no other_changes logged AND the user's own flag.
(`n_clean_events` keeps its name for the Excel contract; it counts usable events.)
The fixed-effects estimate is NOT added as evidence (same data; it would double count);
it's reported alongside as a cross-check.

pooled_betas shares evidence across segments, on ratios to the prior rho_k = beta_k / beta0_k:
    rho_k = lam + d_k,  lam ~ N(1, prior_sd_frac^2 - tau^2),  d_k ~ N(0, tau^2)
    tau = segment_sd_frac
With no events every segment keeps exactly its unpooled prior; events anywhere move lam.
"""

import numpy as np
import pandas as pd

from mortgage_elasticity.model import beta_from_uplift


def posterior_betas(s0: pd.Series, events: pd.DataFrame, u: float,
                    prior_sd_frac: float) -> pd.DataFrame:
    ev = events[events["clean"]] if len(events) else events
    rows = []
    for seg, s in s0.items():
        mu0 = beta_from_uplift(u, s)
        sd0 = prior_sd_frac * mu0
        e = ev[ev["segment"] == seg] if len(ev) else ev
        prec = 1 / sd0**2 + (1 / e["beta_e_se"] ** 2).sum() if len(e) else 1 / sd0**2
        num = mu0 / sd0**2 + (e["beta_e"] / e["beta_e_se"] ** 2).sum() if len(e) else mu0 / sd0**2
        sd, mu = np.sqrt(1 / prec), num / prec
        rows.append({
            "segment": seg, "beta_prior": mu0, "beta_prior_sd": sd0, "n_clean_events": len(e),
            "beta_post": mu, "beta_post_sd": sd,
            "beta_ci90_lo": mu - 1.645 * sd, "beta_ci90_hi": mu + 1.645 * sd,
            "prior_weight": (1 / sd0**2) / prec,  # share of the posterior driven by judgment
        })
    return pd.DataFrame(rows)


def pooled_betas(s0: pd.Series, events: pd.DataFrame, u: float, prior_sd_frac: float,
                 segment_sd_frac: float) -> pd.DataFrame:
    tau2 = segment_sd_frac**2
    var_lam0 = prior_sd_frac**2 - tau2
    b0 = pd.Series([beta_from_uplift(u, s) for s in s0], index=s0.index)
    ev = events[events["clean"]] if len(events) else events
    rho_hat, v = pd.Series(np.nan, index=s0.index), pd.Series(np.inf, index=s0.index)
    if len(ev):
        ev = ev[np.isfinite(ev["beta_e"]) & np.isfinite(ev["beta_e_se"]) & (ev["beta_e_se"] > 0)]
        r = ev["beta_e"] / ev["segment"].map(b0)
        w = (ev["segment"].map(b0) / ev["beta_e_se"]) ** 2  # 1 / se_r^2
        prec = w.groupby(ev["segment"]).sum()
        rho_hat[prec.index] = (r * w).groupby(ev["segment"]).sum() / prec
        v[prec.index] = 1 / prec
    has = np.isfinite(rho_hat)
    lam_prec = 1 / var_lam0 + (1 / (tau2 + v[has])).sum()
    lam = (1 / var_lam0 + (rho_hat[has] / (tau2 + v[has])).sum()) / lam_prec
    own_w = (tau2 / (tau2 + v)).where(has, 0.0)
    mean = own_w * rho_hat.fillna(0) + (1 - own_w) * lam
    var = (tau2 * v / (tau2 + v)).where(has, tau2) + (1 - own_w) ** 2 / lam_prec
    beta, sd = b0 * mean, b0 * np.sqrt(var)
    return pd.DataFrame({
        "segment": s0.index, "beta_pool": beta.to_numpy(), "beta_pool_sd": sd.to_numpy(),
        "beta_pool_ci90_lo": (beta - 1.645 * sd).to_numpy(),
        "beta_pool_ci90_hi": (beta + 1.645 * sd).to_numpy(),
        "u_pool": (beta * (1 - s0)).to_numpy(),
        "shared_scale": lam, "shared_scale_sd": np.sqrt(1 / lam_prec),
        "own_weight": own_w.to_numpy(),
    })
