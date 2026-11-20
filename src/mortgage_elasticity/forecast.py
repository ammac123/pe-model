"""Forecast: what a price change does to share and applications over the next weeks.

Two intervals come back, and they answer different questions:
  price_effect     natural share held where it is today; uncertainty from beta, h and the level
                   estimate only. This is "no competitor response and no market movement".
  full_predictive  natural share also drifts on (the random walks are simulated forward) and
                   lodgement counts are drawn, at trailing-average market volume.
The steady-state multiplier is reported separately as the long-run effect; it is not the answer
for the forecast horizon, because a move takes weeks to phase in.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mortgage_elasticity import panel
from mortgage_elasticity.bayes import Fit
from mortgage_elasticity.model import softmin_rate

VERSIONS = ("price_effect", "full_predictive")


@dataclass
class Paths:
    """Per-draw weekly paths, draws x weeks x segments. Lodgements are at market volume `M`."""
    share_base: np.ndarray  # no change, natural share held
    share_scn: np.ndarray  # with the change, natural share held
    lodg_full: np.ndarray  # with the change: simulated counts with drift in natural share
    long_run: np.ndarray  # draws x segments: steady-state multiplier from a settled baseline
    M: np.ndarray  # weekly market lodgements used (trailing mean), per segment
    conv: np.ndarray  # applications per lodgement (trailing), per segment
    dx: np.ndarray  # scenario change in the unlagged gap, bp, per segment


@dataclass
class Forecast:
    summary: pd.DataFrame  # one row per segment (and the group total) and interval version
    weekly: pd.DataFrame  # the same by week


def _expit(z):
    return 1 / (1 + np.exp(-z))


def benchmark_shift(fit: Fit, competitor_changes: dict | None) -> np.ndarray:
    """Change in r* per segment (bp) if competitors' effective rates change. Keys are a lender
    (applies in every segment) or (segment, lender); values are changes in bp."""
    prep = fit.prep
    out = np.zeros(len(prep.segs))
    if not competitor_changes:
        return out
    if prep.comp is None:
        raise ValueError("competitor_changes given, but the model was fitted without competitors")
    snap = panel.benchmark_snapshot(prep.comp, prep.T, prep.periods, fit.cfg.comp_ffill_periods)
    known = set(snap["lender"]) | set(zip(snap["segment"], snap["lender"]))
    unknown = [k for k in competitor_changes if k not in known]
    if unknown:
        raise ValueError(f"competitor_changes: not in the benchmark: {unknown}")
    for k, seg in enumerate(prep.segs):
        s = snap[snap["segment"] == seg]
        d = np.array([competitor_changes.get((seg, ln), competitor_changes.get(ln, 0.0))
                      for ln in s["lender"]])
        q = s["eff_bps"].to_numpy()
        out[k] = softmin_rate(q + d, prep.T[seg])[0] - softmin_rate(q, prep.T[seg])[0]
    return out


def trailing(fit: Fit) -> tuple[np.ndarray, np.ndarray]:
    """Trailing-window mean weekly market lodgements and conversion ratio, per segment."""
    w = fit.cfg.baseline_window
    conv = panel.conversion_report(fit.prep.df, w).set_index("segment")["conversion"]
    return fit.prep.M[-w:].mean(0), conv[fit.prep.segs].to_numpy()


def paths(fit: Fit, dx, horizon: int | None = None, settled: bool = False,
          seed: int | None = None, median_h: bool = False) -> Paths:
    """dx: change in the unlagged gap from week 1, bp, scalar or per segment.
    settled=True starts from a baseline where past moves have fully phased in, and median_h=True
    holds the half-life at its median: together they are what the Excel export computes."""
    cfg, prep, d = fit.cfg, fit.prep, fit.draws
    horizon = horizon or cfg.horizon
    S, K = fit.n, len(prep.segs)
    dx = np.broadcast_to(np.asarray(dx, float), (K,))
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    beta, level = d["beta"][:, None, :], d["level"][:, -1, :].astype(float)[:, None, :]
    v_T = prep.gap_raw[-1]
    x_T = np.broadcast_to(v_T, (S, K)) if settled else fit.x()[:, -1, :].astype(float)
    # continue the lag recursion: after tau weeks a fraction 1 - (1 - phi)^tau of the way there
    h = np.full(S, np.median(d["h"])) if median_h else d["h"]
    felt = 1 - 2.0 ** (-np.arange(1, horizon + 1) / h[:, None])  # draws x weeks
    x_base = x_T[:, None, :] + (v_T - x_T)[:, None, :] * felt[:, :, None]
    x_scn = x_T[:, None, :] + (v_T + dx - x_T)[:, None, :] * felt[:, :, None]

    # full predictive: random walks simulated forward, then beta-binomial counts
    drift = np.zeros((S, horizon, K))
    if "sigma_g" in d:  # with free week effects there is no trend to project: held flat
        drift += np.cumsum(d["sigma_g"][:, None] * rng.normal(size=(S, horizon)), 1)[:, :, None]
    if "sigma_v" in d:
        z = rng.normal(size=(S, horizon, K))
        drift += np.cumsum(d["sigma_v"][:, None, None] * (z - z.mean(-1, keepdims=True)), 1)
    pi = np.clip(_expit(level + drift - beta * x_scn), 1e-12, 1 - 1e-12)
    M, conv = trailing(fit)
    c = d["c"][:, None, None]
    lodg_full = rng.binomial(np.round(M).astype(int), rng.beta(pi * c, (1 - pi) * c))
    return Paths(share_base=_expit(level - beta * x_base), share_scn=_expit(level - beta * x_scn),
                 lodg_full=lodg_full.astype(float),
                 long_run=(_expit(level - beta * (v_T + dx)) / _expit(level - beta * v_T))[:, 0, :],
                 M=M, conv=conv, dx=dx)


def _totals(p: Paths, idx) -> dict:
    """Per-draw weekly totals over the segments `idx` (lodgements summed within each draw)."""
    M = p.M[idx]
    base, scn = (p.share_base[:, :, idx] * M).sum(-1), (p.share_scn[:, :, idx] * M).sum(-1)
    full = p.lodg_full[:, :, idx].sum(-1)
    # applications = lodgements x conversion, weighted by each segment's scenario lodgements
    apps = lambda lodg: (lodg * p.conv[idx]).sum(-1)
    lr = (p.long_run[:, idx] * p.share_base[:, 0, idx] * M).sum(-1) / base[:, 0]
    return {"M": M.sum(), "base": base, "scn": scn, "full": full, "long_run": lr,
            "apps_base": apps(p.share_base[:, :, idx] * M), "apps_scn": apps(p.share_scn[:, :, idx] * M),
            "apps_full": apps(p.lodg_full[:, :, idx])}


def _q(a, q, prefix: str) -> dict:
    lo, mid, hi = np.quantile(a, q, axis=0)
    return {f"{prefix}_p{round(100 * q[0])}": lo, f"{prefix}_p50": mid,
            f"{prefix}_p{round(100 * q[2])}": hi}


def summarise(p: Paths, idx, q) -> tuple[list[dict], list[dict]]:
    """Summary rows (one per interval version) and weekly rows for one segment or group."""
    t = _totals(p, idx)
    horizon = t["base"].shape[1]
    base_lodg = np.median(t["base"].sum(1))  # full-predictive multipliers are against this
    rows, weekly = [], []
    for version, lodg, apps in [("price_effect", t["scn"], t["apps_scn"]),
                                ("full_predictive", t["full"], t["apps_full"])]:
        mult = (lodg.sum(1) / (t["base"].sum(1) if version == "price_effect" else base_lodg))
        rows.append({"version": version, "dx_bps": float(np.mean(p.dx[idx])),
                     **_q(mult, q, "multiplier"),
                     **_q(100 * lodg.sum(1) / (horizon * t["M"]), q, "share_pp"),
                     **_q(apps.sum(1), q, "apps"),
                     "baseline_apps_p50": np.median(t["apps_base"].sum(1)),
                     "baseline_share_pp_p50": 100 * base_lodg / (horizon * t["M"]),
                     **(_q(t["long_run"], q, "long_run_multiplier")
                        if version == "price_effect" else {})})
        wk_mult = lodg / (t["base"] if version == "price_effect" else np.median(t["base"], 0))
        wm, ws = _q(wk_mult, q, "multiplier"), _q(100 * lodg / t["M"], q, "share_pp")
        weekly += [{"version": version, "week": w + 1, **{k: v[w] for k, v in {**wm, **ws}.items()}}
                   for w in range(horizon)]
    return rows, weekly


def forecast(fit: Fit, segments, d_headline_bps: float, d_discretionary_bps: float = 0.0,
             competitor_changes: dict | None = None, horizon: int | None = None,
             settled: bool = False, median_h: bool = False) -> Forecast:
    """Effect over `horizon` weeks of changing our rate in `segments` (names, or "ALL").
    With several segments a TOTAL row sums their lodgements within each draw.
    settled and median_h: see `paths`."""
    prep = fit.prep
    segments = prep.segs if isinstance(segments, str) and segments.upper() == "ALL" else (
        [segments] if isinstance(segments, str) else list(segments))
    unknown = [s for s in segments if s not in prep.segs]
    if unknown:
        raise ValueError(f"forecast: unknown segments {unknown}")
    idx = [prep.segs.index(s) for s in segments]
    dx = -benchmark_shift(fit, competitor_changes)
    dx[idx] += d_headline_bps + prep.kappa * d_discretionary_bps
    p = paths(fit, dx, horizon, settled, median_h=median_h)
    units = [(s, [i]) for s, i in zip(segments, idx)] + ([("TOTAL", idx)] if len(idx) > 1 else [])
    rows, weekly = [], []
    for name, ii in units:
        r, w = summarise(p, ii, fit.cfg.q)
        rows += [{"segment": name, **x} for x in r]
        weekly += [{"segment": name, **x} for x in w]
    return Forecast(pd.DataFrame(rows), pd.DataFrame(weekly))


def groups(fit: Fit) -> list[tuple[str, str, list[int]]]:
    """(kind, name, segment indices) for every segment and for the reporting groups."""
    dims = fit.prep.dims.reset_index()
    out = [("segment", s, [i]) for i, s in enumerate(fit.prep.segs)]
    out.append(("all", "ALL", list(range(len(dims)))))
    for kind in ("purpose", "repayment", "lvr_band"):
        out += [(kind, name, list(g.index)) for name, g in dims.groupby(kind, sort=False)]
    return out


def accuracy_table(fit: Fit) -> pd.DataFrame:
    """What the tool can promise: the half-width of the interval of horizon-total applications
    and share, per segment and group and per own headline move, against the business targets.
    Nothing is tuned to make this pass."""
    cfg = fit.cfg
    lo, hi = f"p{round(100 * cfg.q[0])}", f"p{round(100 * cfg.q[2])}"
    rows = []
    for move in cfg.accuracy_moves_bps:
        p = paths(fit, move)
        for kind, name, idx in groups(fit):
            row = {"kind": kind, "name": name, "move_bps": move}
            for r in summarise(p, idx, cfg.q)[0]:
                v = r["version"]
                row[f"apps_p50_{v}"] = r["apps_p50"]
                row[f"apps_halfwidth_rel_{v}"] = (r[f"apps_{hi}"] - r[f"apps_{lo}"]) / 2 / r["apps_p50"]
                row[f"share_halfwidth_pp_{v}"] = (r[f"share_pp_{hi}"] - r[f"share_pp_{lo}"]) / 2
                row[f"meets_5pct_{v}"] = bool(row[f"apps_halfwidth_rel_{v}"] <= cfg.target_apps_rel)
                row[f"meets_0p3pp_{v}"] = bool(row[f"share_halfwidth_pp_{v}"] <= cfg.target_share_pp)
            rows.append(row)
    return pd.DataFrame(rows)
