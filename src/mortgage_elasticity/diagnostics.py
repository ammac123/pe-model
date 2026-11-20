"""Diagnostics: tables and plots only. None of this feeds the estimate.

  raw_move_check   no model at all: did applications move after past price moves the way an
                   assumed uplift says they should? Run it before any fit.
  estimators       two-way fixed effects and event-study estimates beside the model's u_k, each
                   with its known bias (docs/model_maths.tex, section 8).
  ppc              posterior predictive checks: does simulated share wobble and persist from
                   week to week the way the actual share does?
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from mortgage_elasticity import events, fe, panel, plots
from mortgage_elasticity.bayes import Fit
from mortgage_elasticity.model import multiplier
from mortgage_elasticity.panel import Prepared
from mortgage_elasticity.runlog import log

TWFE_BIAS = ("towards zero when our moves react to share or competitor discounts are unseen; "
             "week effects absorb every bank-wide move; its standard error is too small when "
             "share drifts")
EVENT_BIAS = ("towards zero for moves made after share had slipped; bank-wide moves have no "
              "comparison segments and are dropped; few events, so noisy")


def raw_move_check(prep: Prepared, cfg, out_dir: Path | None = None) -> pd.DataFrame:
    """Per own price move and affected segment (plus a TOTAL row): the change in the unlagged gap,
    applications and share before and after, their ratio, and the ratio each uplift in
    cfg.check_u implies. Writes one table and one plot per move into out_dir."""
    pre, (lo, hi) = cfg.move_pre_weeks, cfg.move_post_weeks
    n = len(prep.periods)
    apps, labels = prep.wide("apps"), [f"u={u:.0%}/bp" for u in cfg.check_u]
    rows = []
    for i, ev in enumerate(prep.moves("price").itertuples(), start=1):
        before, after = slice(max(ev.t - pre, 0), ev.t), slice(ev.t + lo, min(ev.t + hi, n - 1) + 1)
        if ev.t - pre < 0 or ev.t + lo >= n:
            log(f"check-moves: move {i} ({ev.period.date()}) is too close to the edge of the "
                "panel, skipped")
            continue
        idx = [prep.segs.index(s) for s in ev.segments]
        move = {"move": i, "date": ev.period.date(), "description": ev.description,
                "bank_wide": ev.all, "post_weeks": after.stop - after.start}
        part = []
        for k in idx:
            share = [prep.L[w, k].sum() / max(prep.M[w, k].sum(), 1) for w in (before, after)]
            dgap = prep.gap_raw[after, k].mean() - prep.gap_raw[before, k].mean()
            part.append({**move, "segment": prep.segs[k], "d_gap_bps": dgap,
                         "apps_before": apps[before, k].mean(), "apps_after": apps[after, k].mean(),
                         "share_before": share[0], "share_after": share[1],
                         **{f"implied_ratio_{lab}": float(multiplier(dgap, share[0], u / (1 - share[0])))
                            for lab, u in zip(labels, cfg.check_u)}})
        part = pd.DataFrame(part)
        w = part["apps_before"]  # the total is the applications-weighted view of the segments
        total = {**move, "segment": "TOTAL", "d_gap_bps": np.average(part["d_gap_bps"], weights=w)
                 if w.sum() else np.nan,
                 "apps_before": w.sum(), "apps_after": part["apps_after"].sum(),
                 "share_before": prep.L[before][:, idx].sum() / max(prep.M[before][:, idx].sum(), 1),
                 "share_after": prep.L[after][:, idx].sum() / max(prep.M[after][:, idx].sum(), 1),
                 **{f"implied_ratio_{lab}": (part[f"implied_ratio_{lab}"] * w).sum() / w.sum()
                    if w.sum() else np.nan for lab in labels}}
        part = pd.concat([part, pd.DataFrame([total])], ignore_index=True)
        part["apps_ratio"] = part["apps_after"] / part["apps_before"]
        part["share_ratio"] = part["share_after"] / part["share_before"]
        rows.append(part)
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = f"move_{i:02d}_{ev.period.date()}"
            part.to_csv(out_dir / f"{stem}.csv", index=False)
            weeks = np.arange(-pre, min(hi, n - 1 - ev.t) + 1)
            series = apps[ev.t + weeks][:, idx].sum(1) / w.sum() if w.sum() else weeks * np.nan
            what = "bank-wide" if ev.all else f"{len(idx)} segment" + "s" * (len(idx) > 1)
            plots.raw_move(out_dir / f"{stem}.png",
                           f"Move {i}, {ev.period.date()}: {ev.description} ({what}, gap "
                           f"{total['d_gap_bps']:+.1f}bp)", weeks, series,
                           {lab: total[f"implied_ratio_{lab}"] for lab in labels}, pre, (lo, hi))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _event_rows(prep: Prepared, cfg) -> pd.DataFrame:
    """The event log in the shape events.event_study wants: one row per price move and segment,
    with any non-price change logged for that segment inside the window."""
    other, rows = prep.moves("other"), []
    for ev in prep.moves("price").itertuples():
        for seg in ev.segments:
            near = other[np.array([seg in s for s in other["segments"]], bool)
                         & (other["t"] >= ev.t - cfg.event_pre).to_numpy()
                         & (other["t"] < ev.t + cfg.event_post).to_numpy()]
            rows.append({"segment": seg, "period": ev.period, "reason": ev.description,
                         "clean": True, "other_changes": "; ".join(near["description"])})
    return pd.DataFrame(rows, columns=["segment", "period", "reason", "clean", "other_changes"])


def estimators(fit: Fit) -> pd.DataFrame:
    """The model's u_k beside the two-way fixed effects and event-study estimates."""
    prep, cfg, q = fit.prep, fit.cfg, fit.cfg.q
    out = pd.DataFrame({"segment": prep.segs, "s0": prep.s0})
    lo, mid, hi = fit.q("u")
    out[f"u_model_p{round(100 * q[0])}"], out["u_model_p50"] = lo, mid
    out[f"u_model_p{round(100 * q[2])}"] = hi
    df = panel.add_perceived_price(prep.df, float(np.median(fit.draws["h"])))
    ok = df.groupby("segment")["share"].agg(lambda s: bool(((s > 0) & (s < 1)).all()))
    if not ok.all():
        log(f"diagnostics: {int((~ok).sum())} segments have weeks with zero share or no market, "
            "so no logit share: left out of the fixed-effects and event-study checks")
    df = df[df["segment"].map(ok)].copy()
    out["u_twfe"], out["u_twfe_se"], out["u_event"], out["n_events"] = np.nan, np.nan, np.nan, 0
    if df["segment"].nunique() > 1:
        df["y"] = np.log(df["share"] / (1 - df["share"]))
        try:
            b = fe.betas_from_fit(fe.fit_twfe(df)).set_index("segment")
            out["u_twfe"] = out["segment"].map(b["beta_fe"]) * (1 - out["s0"])
            out["u_twfe_se"] = out["segment"].map(b["beta_fe_se"]) * (1 - out["s0"])
        except ValueError as exc:
            log(f"diagnostics: two-way fixed effects not estimable ({exc})")
        ev = _event_rows(prep, cfg)
        ev = ev[ev["segment"].isin(df["segment"])]
        with warnings.catch_warnings():  # few estimable events is expected; the summary says so
            warnings.simplefilter("ignore")
            es, summary = events.event_study(df, ev, cfg.event_pre, cfg.event_post, cfg.min_dx_bps)
        log(f"diagnostics: {summary}")
        es = es[es["clean"]]
        if len(es):  # precision-weighted mean of the segment's usable events
            w = 1 / es["beta_e_se"] ** 2
            b = (es["beta_e"] * w).groupby(es["segment"]).sum() / w.groupby(es["segment"]).sum()
            out["u_event"] = out["segment"].map(b) * (1 - out["s0"])
            out["n_events"] = out["segment"].map(es.groupby("segment").size()).fillna(0).astype(int)
    out["twfe_known_bias"], out["event_known_bias"] = TWFE_BIAS, EVENT_BIAS
    return out


def _wobble(share: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Variance and lag-1 autocorrelation of weekly share along the week axis (-2)."""
    d = share - np.nanmean(share, axis=-2, keepdims=True)
    var = np.nanmean(d**2, axis=-2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return var, np.nanmean(d[..., 1:, :] * d[..., :-1, :], axis=-2) / var


def ppc(fit: Fit) -> pd.DataFrame:
    """Weekly share variance and lag-1 autocorrelation per segment: observed against simulated."""
    prep, cfg = fit.prep, fit.cfg
    rng = np.random.default_rng(cfg.seed)
    pick = rng.choice(fit.n, size=min(cfg.ppc_draws, fit.n), replace=False)
    pi = np.clip(fit.pi()[pick], 1e-12, 1 - 1e-12)
    c = fit.draws["c"][pick, None, None]
    M = np.where(prep.obs, prep.M, np.nan)
    with np.errstate(invalid="ignore"):
        sim = rng.binomial(prep.M, rng.beta(pi * c, (1 - pi) * c)) / M
    rows = []
    for stat, obs, s in zip(("share_variance", "lag1_autocorrelation"), _wobble(prep.L / M),
                            _wobble(sim)):
        lo, mid, hi = np.nanquantile(s, cfg.q, axis=0)
        rows.append(pd.DataFrame({"segment": prep.segs, "statistic": stat, "observed": obs,
                                  "simulated_low": lo, "simulated_p50": mid, "simulated_high": hi,
                                  "inside": (obs >= lo) & (obs <= hi)}))
    return pd.concat(rows, ignore_index=True)
