"""Validation and sensitivity runs. Each is a refit, so re-validation is a rerun.

  events       leave one own price move out: mask the 12 weeks after it in the affected segments,
               refit, and predict those weeks from the market volumes and gap path that occurred.
  quiet        rolling-origin 12-week forecasts on segments we did not reprice: the error floor
               from drift and sampling.
  placebo      a fake move at a quiet date should show no effect.
  sensitivity  how far u moves with the prior, T, kappa, the time effect and the jumps.

Refits use the soft-min temperature of the base fit (no T loop) and the val_* sampler settings.
"""

from dataclasses import replace

import numpy as np
import pandas as pd

from mortgage_elasticity import bayes
from mortgage_elasticity.bayes import Fit
from mortgage_elasticity.forecast import groups
from mortgage_elasticity.runlog import log


def _refit(base: Fit, label: str, cfg=None, prep=None, **kw) -> Fit:
    cfg = cfg or base.cfg
    return bayes.fit(prep or base.prep, cfg, chains=cfg.val_chains, tune=cfg.val_tune,
                     draws=cfg.val_draws, label=label, check=False, **kw)


def _predict(fit: Fit, weeks: slice, rng) -> np.ndarray:
    """Simulated lodgements per draw for `weeks` (draws x weeks x segments), at actual volumes."""
    pi = np.clip(fit.pi()[:, weeks], 1e-12, 1 - 1e-12)
    c = fit.draws["c"][:, None, None]
    return rng.binomial(fit.prep.M[weeks], rng.beta(pi * c, (1 - pi) * c)).astype(float)


def _score(sim: np.ndarray, actual: float, q) -> dict:
    """sim: per-draw totals. Predicted quantiles against the actual total."""
    lo, mid, hi = np.quantile(sim, q)
    return {"predicted_low": lo, "predicted_p50": mid, "predicted_high": hi, "actual": actual,
            "rel_error": mid / actual - 1 if actual else np.nan,
            "covered": bool(lo <= actual <= hi)}


def _coverage(rows: pd.DataFrame, what: str, interval: float) -> None:
    for unit, g in rows.groupby(rows["segment"].eq("TOTAL").map({True: "totals", False: "segments"})):
        log(f"validate {what}: {interval:.0%} interval covered the actual in {g['covered'].mean():.0%} "
            f"of {len(g)} {unit}; median absolute error {g['rel_error'].abs().median():.1%}")


def events(base: Fit) -> pd.DataFrame:
    prep, cfg = base.prep, base.cfg
    n, rng, rows = len(prep.periods), np.random.default_rng(cfg.seed), []
    for i, ev in enumerate(prep.moves("price").itertuples(), start=1):
        weeks = slice(ev.t, min(ev.t + cfg.horizon, n))
        idx = [prep.segs.index(s) for s in ev.segments]
        mask = prep.obs.copy()
        mask[weeks, idx] = False
        sim = _predict(_refit(base, f"validate events, move {i}", mask=mask), weeks, rng)
        sim, actual = sim * prep.obs[weeks], prep.L[weeks] * prep.obs[weeks]
        info = {"move": i, "date": ev.period.date(), "description": ev.description,
                "bank_wide": ev.all, "weeks": weeks.stop - weeks.start}
        rows.append({**info, "segment": "TOTAL",
                     **_score(sim[:, :, idx].sum((1, 2)), actual[:, idx].sum(), cfg.q)})
        rows += [{**info, "segment": prep.segs[k],
                  **_score(sim[:, :, k].sum(1), actual[:, k].sum(), cfg.q)} for k in idx]
    out = pd.DataFrame(rows)
    if len(out):
        _coverage(out, "events", cfg.interval)
    return out


def quiet(base: Fit) -> pd.DataFrame:
    prep, cfg = base.prep, base.cfg
    n, rng, rows = len(prep.periods), np.random.default_rng(cfg.seed), []
    price, H = prep.moves("price"), prep.wide("headline_cum_bps")
    for t0 in range(cfg.quiet_first_origin, n - cfg.horizon + 1, cfg.quiet_step):
        weeks = slice(t0, t0 + cfg.horizon)
        moved = {s for ev in price.itertuples() if t0 <= ev.t < weeks.stop for s in ev.segments}
        idx = [k for k, s in enumerate(prep.segs)
               if s not in moved and np.ptp(H[t0 - 1:weeks.stop, k]) == 0]
        if not idx:
            log(f"validate quiet: origin week {t0}: every segment was repriced, skipped")
            continue
        mask = prep.obs.copy()
        mask[t0:] = False
        sim = _predict(_refit(base, f"validate quiet, origin week {t0}", mask=mask), weeks, rng)
        sim, actual = sim * prep.obs[weeks], prep.L[weeks] * prep.obs[weeks]
        info = {"origin_week": t0, "origin": prep.periods[t0 - 1].date(), "n_quiet": len(idx)}
        rows.append({**info, "segment": "TOTAL",
                     **_score(sim[:, :, idx].sum((1, 2)), actual[:, idx].sum(), cfg.q)})
        rows += [{**info, "segment": prep.segs[k],
                  **_score(sim[:, :, k].sum(1), actual[:, k].sum(), cfg.q)} for k in idx]
    out = pd.DataFrame(rows)
    if len(out):
        _coverage(out, "quiet", cfg.interval)
    return out


def placebo(base: Fit) -> pd.DataFrame:
    """A fake step of cfg.placebo_bps at a random quiet date and segment (no own move within
    cfg.event_post weeks), with its own coefficient. Its interval should contain zero about as
    often as the interval level."""
    prep, cfg = base.prep, base.cfg
    n, K = prep.shape
    rng, price, rows = np.random.default_rng(cfg.seed), prep.moves("price"), []
    quiet_cells = [(t, k) for k, s in enumerate(prep.segs)
                   for t in range(cfg.horizon + 1, n - cfg.horizon)
                   if all(abs(ev.t - t) > cfg.event_post for ev in price.itertuples()
                          if s in ev.segments)]
    if not quiet_cells:
        log("validate placebo: no quiet date in any segment")
        return pd.DataFrame()
    for i in range(1, cfg.placebo_runs + 1):
        t, k = quiet_cells[rng.integers(len(quiet_cells))]
        step = np.zeros((n, K))
        step[t:, k] = cfg.placebo_bps
        coef = _refit(base, f"validate placebo {i}", placebo_step=step).draws["placebo"]
        lo, mid, hi = np.quantile(coef, cfg.q)
        rows.append({"run": i, "segment": prep.segs[k], "date": prep.periods[t].date(),
                     "fake_move_bps": cfg.placebo_bps, "coef_low": lo, "coef_p50": mid,
                     "coef_high": hi, "contains_zero": bool(lo <= 0 <= hi),
                     "segment_beta_p50": float(np.median(base.draws["beta"][:, k]))})
    out = pd.DataFrame(rows)
    log(f"validate placebo: the {cfg.interval:.0%} interval of the fake coefficient contains zero "
        f"in {out['contains_zero'].mean():.0%} of {len(out)} runs (expect about {cfg.interval:.0%})")
    return out


def group_u(fit: Fit, run: str) -> pd.DataFrame:
    """u per segment group: the lodgement-weighted mean of u_k within each draw."""
    w, rows = fit.prep.L.mean(0), []
    for kind, name, idx in groups(fit):
        if kind == "segment":
            continue
        lo, mid, hi = np.quantile(fit.draws["u"][:, idx] @ (w[idx] / w[idx].sum()), fit.cfg.q)
        rows.append({"run": run, "kind": kind, "name": name, "u_low": lo, "u_p50": mid,
                     "u_high": hi})
    return pd.DataFrame(rows)


def sensitivity(base: Fit) -> pd.DataFrame:
    """One table of u (low, middle, high) overall and by group for every run, beside the base."""
    cfg, prep = base.cfg, base.prep
    runs = [(f"prior median {m:.0%}/bp", replace(cfg, prior_u_median=m), prep)
            for m in cfg.sens_prior_medians]
    if prep.comp is not None:
        runs += [(f"T x {f:g}", cfg, prep.with_T((prep.T * f).clip(*cfg.T_bounds), cfg))
                 for f in cfg.sens_T_factors]
    runs += [(f"kappa {k:g}", replace(cfg, kappa=k), prep.with_kappa(k)) for k in cfg.sens_kappas]
    runs.append(("week dummies (time_effect=fe)", replace(cfg, time_effect="fe"), prep))
    runs.append(("jumps off", replace(cfg, jumps=False), prep))
    out = [group_u(base, "base")]
    for label, c, p in runs:
        out.append(group_u(_refit(base, f"sensitivity, {label}", cfg=c, prep=p), label))
    out = pd.concat(out, ignore_index=True)
    out["u_p50_vs_base"] = out["u_p50"] / out.groupby(["kind", "name"])["u_p50"].transform("first") - 1
    return out
