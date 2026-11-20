"""Outputs of `estimate`: the csv tables and the natural-share charts."""

from pathlib import Path

import numpy as np
import pandas as pd

from mortgage_elasticity import panel, plots
from mortgage_elasticity.bayes import GLOBAL, Fit


def _cols(prefix: str, q) -> list[str]:
    return [f"{prefix}_p{round(100 * q[0])}", f"{prefix}_p50", f"{prefix}_p{round(100 * q[2])}"]


def prior_share(fit: Fit) -> np.ndarray:
    """Var_posterior(log u_k) / Var_prior(log u_k), clipped to [0, 1]: near 1 = mostly prior."""
    return np.clip(np.log(fit.draws["u"]).var(0) / fit.prior_log_u.var(0), 0, 1)


def share_now(fit: Fit) -> np.ndarray:
    """Fitted share in the last week, per draw: the forecast baseline at week 0."""
    eta = fit.draws["level"][:, -1, :] - fit.draws["beta"] * fit.x()[:, -1, :]
    return 1 / (1 + np.exp(-eta))


def segments_table(fit: Fit) -> pd.DataFrame:
    prep, cfg, q = fit.prep, fit.cfg, fit.cfg.q
    out = prep.dims.reset_index()
    out["lodg_us_per_week"], out["lodg_mkt_per_week"] = prep.L.mean(0), prep.M.mean(0)
    for name, draws in [("share_now", share_now(fit)), ("u", fit.draws["u"]),
                        ("beta", fit.draws["beta"])]:
        out[_cols(name, q)] = np.quantile(draws, q, axis=0).T
    out["T_bps"] = np.nan if prep.T is None else prep.T[prep.segs].to_numpy()
    price = prep.moves("price")
    out["n_price_moves"] = [int(sum(s in e for e in price["segments"])) for s in prep.segs]
    out["prior_share"] = prior_share(fit)
    out = out.merge(panel.conversion_report(prep.df, cfg.baseline_window), on="segment")
    out["thin_volume"] = out["lodg_us_per_week"] < cfg.thin_lodgements
    out["u_rhat"], out["u_ess"] = fit.diag["u_rhat"], fit.diag["u_ess"]
    out["sampler_warning"] = (out["u_rhat"] > cfg.max_rhat) | (out["u_ess"] < cfg.min_ess) | (
        fit.diag["divergences"] > 0)
    return out


def global_table(fit: Fit) -> pd.DataFrame:
    q, rows = fit.cfg.q, []
    for name in GLOBAL:
        if name not in fit.draws:  # e.g. sigma_g with free week effects
            continue
        d = fit.draws[name]
        if name == "mu_u":  # reported as the typical uplift, % per bp
            name, d = "mu_u_pct_per_bp", 100 * np.exp(d)
        labels = [f"{name}[{b}]" for b in fit.cfg.lvr_bands] if d.ndim == 2 else [name]
        for label, col in zip(labels, np.atleast_2d(d.T)):
            rows.append({"parameter": label, **dict(zip(_cols("", q), np.quantile(col, q)))})
    return pd.DataFrame(rows).rename(columns=lambda c: c.lstrip("_"))


def natural_share_table(fit: Fit) -> pd.DataFrame:
    prep, q = fit.prep, fit.cfg.q
    natural = np.quantile(1 / (1 + np.exp(-fit.draws["level"].astype(float))), q, axis=0)
    fitted = np.median(fit.pi(), axis=0)
    n, K = prep.shape
    out = pd.DataFrame({"segment": np.tile(prep.segs, n), "period": np.repeat(prep.periods, K),
                        "share_obs": (prep.L / np.where(prep.M > 0, prep.M, np.nan)).ravel(),
                        "share_fitted_p50": fitted.ravel()})
    out[_cols("natural_share", q)] = natural.reshape(3, -1).T
    return out.sort_values(["segment", "period"], kind="stable").reset_index(drop=True)


def natural_share_plots(fit: Fit, table: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = _cols("natural_share", fit.cfg.q)
    for seg, g in table.groupby("segment", sort=False):
        ev = fit.prep.events[[seg in s for s in fit.prep.events["segments"]]]
        plots.natural_share(out_dir / f"{plots.filename(seg)}.png", seg, g["period"],
                            g["share_obs"], g["share_fitted_p50"].to_numpy(),
                            g[cols].to_numpy().T, ev, fit.cfg.interval)
