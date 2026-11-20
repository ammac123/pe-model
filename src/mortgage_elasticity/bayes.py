"""Step 5b (optional): full hierarchical PANEL model in PyMC.  `uv sync --extra bayes`
The main estimate when competitor data is available (beta_fe is the cross-check).

    y_kt ~ N(alpha_k + w_kt + delta_t - beta_k * x_kt, sigma)
    w_kt = q * sum_{s<=t} eps_ks,  eps_.s ~ ZeroSumNormal over segments  (drifting level, as alpha.py)
    beta_k ~ N(mu_b, tau),  mu_b ~ N(u / (1 - mean s0), prior_sd_frac * that)

The drifting level matters: with a fixed alpha_k, slow drift in share reads as certainty and the
90% intervals covered the true beta ~17% of the time in simulation (~80-92% with the drift).

kappa and the half-lives are held at the grid-search values (x precomputed). This fits the whole
panel like beta_fe and never sees the event log. Moves made because share had already fallen are
measured against the drifted level, so they don't bias it the way they bias beta_fe (simulation:
0.95-1.12 x truth vs 0.77-1.11 for beta_fe on the gap path). Moves made on signals not yet in share still would.
"""

import pandas as pd


def fit_hierarchical(df: pd.DataFrame, s0: pd.Series, u: float, prior_sd_frac: float,
                     draws: int = 1000, tune: int = 1000, seed: int = 0) -> pd.DataFrame:
    try:
        import warnings

        import arviz as az
        import numpy as np
        import pymc as pm
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("PyMC not installed. Run: uv sync --extra bayes") from exc

    segs = list(s0.index)
    seg_idx = df["segment"].map({s: i for i, s in enumerate(segs)}).to_numpy()
    t_codes, _ = pd.factorize(df["period"])
    n_t, n_k = int(t_codes.max()) + 1, len(segs)
    mu_prior = u / (1 - s0.mean())

    with pm.Model():
        mu_b = pm.Normal("mu_b", mu=mu_prior, sigma=prior_sd_frac * mu_prior)
        tau = pm.HalfNormal("tau", sigma=0.5 * mu_prior)
        beta = pm.Normal("beta", mu=mu_b, sigma=tau, shape=len(segs))
        alpha = pm.Normal("alpha", 0, 3, shape=len(segs))
        # sum-to-zero period effects: identified against alpha (a free mean would never converge)
        delta = pm.ZeroSumNormal("delta", sigma=0.5, shape=n_t)
        # segment drift, zero-sum across segments so common drift stays in delta
        q = pm.HalfNormal("q", 0.05)
        eps = pm.ZeroSumNormal("eps", sigma=1.0, shape=(n_t, n_k))
        w = pm.math.cumsum(eps, axis=0) * q
        sigma = pm.HalfNormal("sigma", 0.5)
        mu_y = (alpha[seg_idx] + w[t_codes, seg_idx] + delta[t_codes]
                - beta[seg_idx] * df["x"].to_numpy())
        pm.Normal("y", mu=mu_y, sigma=sigma, observed=df["y"].to_numpy())
        idata = pm.sample(draws=draws, tune=tune, random_seed=seed, progressbar=False,
                          target_accept=0.9)

    diag = az.summary(idata, var_names=["beta"], kind="diagnostics")
    n_div = int(idata.sample_stats["diverging"].sum())
    if diag["r_hat"].max() > 1.01 or diag["ess_bulk"].min() < 400 or n_div:
        warnings.warn(f"PyMC sampling looks unreliable: max r_hat {diag['r_hat'].max():.3f}, "
                      f"min ess {diag['ess_bulk'].min():.0f}, {n_div} divergences")
    b = idata.posterior["beta"].values.reshape(-1, len(segs))  # (chains*draws, segments)
    return pd.DataFrame({
        "segment": segs, "mean": b.mean(0), "sd": b.std(0),
        "ci90_lo": np.quantile(b, 0.05, axis=0), "ci90_hi": np.quantile(b, 0.95, axis=0),
        "r_hat": diag["r_hat"].to_numpy(), "ess_bulk": diag["ess_bulk"].to_numpy(),
        "divergences": n_div,
    })
