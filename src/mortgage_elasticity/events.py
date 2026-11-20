"""Diagnostic only: difference-in-differences event studies on logit share.
Nothing here feeds the estimate (bayes.py); diagnostics.py reports it beside the model's u_k.

For an event in segment k at period t_e:
    dy  = (mean y_post - mean y_pre)_k  -  same for control segments (no event in the window)
    dx  = (mean x_post - mean x_pre)_k  -  same for controls, using the lagged perceived price
    beta_e = -dy / dx

`period` is the period END: an event dated mid-period lands in the first panel period >= its date.

Rows are one own price move in one segment (diagnostics.py builds them from the event log).
`clean` = no logged non-price change in the window (`other_changes`).
"""

import warnings

import numpy as np
import pandas as pd

COLS = ["segment", "period", "event_date", "reason", "clean", "n_controls", "own_events_in_window",
        "dx_bps", "dx_controls_bps", "dy_logit", "beta_e", "beta_e_se", "other_changes",
        "flagged_clean"]
REASONS = ["after_end", "window_edge", "no_controls", "small_dx", "non_finite"]


def _change(w: np.ndarray, lo: int, pre: int, post: int) -> np.ndarray:
    """Post-mean minus pre-mean for every column of a periods x segments block."""
    return w[lo + pre:lo + pre + post].mean(0) - w[lo:lo + pre].mean(0)


def _placebo_sd(y, x, busy, pre, post, min_dx):
    """Robust sd of the DiD dy over windows with no event and |dx| < min_dx, measured against all
    K-1 other segments. Picks up autocorrelation / drift that period-level noise misses."""
    K = y.shape[1]
    dys = []
    for lo in range(len(y) - pre - post + 1):
        cy, cx = _change(y, lo, pre, post), _change(x, lo, pre, post)
        dy = cy - (cy.sum() - cy) / (K - 1)
        dx = cx - (cx.sum() - cx) / (K - 1)
        ok = (np.abs(dx) < min_dx) & ~busy[lo:lo + pre + post].any(0)
        dys.extend(dy[ok])
    dys = np.asarray(dys)
    if len(dys) < 10:
        return None, len(dys)
    return 1.4826 * np.median(np.abs(dys - np.median(dys))), len(dys)


def _noise(y, x, busy, pre, post, min_dx):
    """Placebo sd of the (pre, post) DiD dy; first-difference fallback if too few windows."""
    K = y.shape[1]
    sd, n = _placebo_sd(y, x, busy, pre, post, min_dx) if K > 1 else (None, 0)
    if sd is None:  # fallback: period-level noise from first differences
        sd = np.nanstd(np.diff(y, axis=0), ddof=1) / np.sqrt(2) * np.sqrt(1 / pre + 1 / post)
        sd *= np.sqrt(1 + 1 / max(K - 1, 1))  # put on the same K-1-controls footing
    return sd, n


def event_study(df: pd.DataFrame, events: pd.DataFrame, pre: int, post: int,
                min_dx: float = 1.0, beta_sd: pd.Series | None = None) -> tuple[pd.DataFrame, str]:
    """beta_sd (per segment, optional) adds the heterogeneity term: beta_e is off by
    (beta_k - beta_c) * dx_c / dx when the controls' x also moved."""
    periods = np.sort(df["period"].unique())
    wide_y = df.pivot(index="period", columns="segment", values="y")
    wide_x = df.pivot(index="period", columns="segment", values="x")
    segs = list(wide_y.columns)
    y, x = wide_y.to_numpy(), wide_x.to_numpy()
    i = np.searchsorted(periods, events["period"].to_numpy(), side="left")
    ev = events.assign(event_date=events["period"], i=i)[i < len(periods)]
    busy = np.zeros(y.shape, bool)
    busy[ev["i"].to_numpy(), [segs.index(s) for s in ev["segment"]]] = True

    sd, n_placebo = _noise(y, x, busy, pre, post, min_dx)

    dropped = dict.fromkeys(REASONS, 0)
    dropped["after_end"] = len(events) - len(ev)
    rows, K = [], len(segs)
    for _, e in ev.iterrows():
        i, seg = int(e["i"]), e["segment"]
        lo, hi = i - pre, i + post
        if lo < 0 or hi > len(periods):
            dropped["window_edge"] += 1
            continue
        in_win = ev[(ev["i"] >= lo) & (ev["i"] < hi)]
        controls = [segs.index(c) for c in segs if c not in set(in_win["segment"])]
        if not controls:
            dropped["no_controls"] += 1
            continue
        k = segs.index(seg)
        cy, cx = _change(y, lo, pre, post), _change(x, lo, pre, post)
        dy, dx_c = cy[k] - cy[controls].mean(), cx[controls].mean()
        dx = cx[k] - dx_c
        if abs(dx) < min_dx:
            dropped["small_dx"] += 1
            continue
        nc = len(controls)
        se_dy = sd * np.sqrt(1 + 1 / nc) / np.sqrt(1 + 1 / max(K - 1, 1))
        b_sd = beta_sd[seg] if beta_sd is not None else 0.0
        beta_e, se = -dy / dx, np.hypot(se_dy / abs(dx), b_sd * abs(dx_c) / abs(dx))
        if not (np.isfinite([dy, dx, beta_e, se]).all() and se > 0):
            dropped["non_finite"] += 1
            continue
        other = str(e.get("other_changes", ""))
        other = "" if other == "nan" else other
        rows.append({
            "segment": seg, "period": periods[i], "event_date": e["event_date"],
            "reason": e["reason"], "clean": bool(not other and e["clean"]),
            "n_controls": nc,
            "own_events_in_window": int((in_win["segment"] == seg).sum()) - 1,
            "dx_bps": dx, "dx_controls_bps": dx_c, "dy_logit": dy,
            "beta_e": beta_e, "beta_e_se": se,
            "other_changes": other, "flagged_clean": bool(e["clean"]),
        })
    es = pd.DataFrame(rows, columns=COLS)
    summary = (f"Events: {len(events)} given, {len(es)} estimable; dropped: "
               + ", ".join(f"{r}={n}" for r, n in dropped.items())
               + f" (noise from {n_placebo} placebo windows"
               + (")" if n_placebo >= 10 else "; too few, used first differences)"))
    if len(es) == 0 or len(es) < len(events) / 2:
        warnings.warn(f"only {len(es)} of {len(events)} events estimable")
    return es, summary
