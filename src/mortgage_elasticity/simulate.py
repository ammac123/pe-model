"""Synthetic panel + event log with known parameters, for testing the pipeline end to end.

competitors=True: every segment also has 6-8 lenders. RBA-like steps are passed on by everyone
(us included), each lender after its own 0-3 period delay; lenders also run temporary specials.
Truth is the gap model x = lag(H) + kappa lag(D) - lag(r* - r*_0), r* = softmin(eff, 1/beta_k).
reactive=True: we also cut 10bp whenever volume falls short, and the natural position drifts more,
so shortfalls persist; beta_fe is then biased towards zero. These cuts are logged clean=False, as if
a user who could tell them apart had vetoed them. Real logs won't be so tidy: set them TRUE (or drop
the column) and beta_post / beta_pool come out about 17% low (tests/test_pipeline.py).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from mortgage_elasticity.model import softmin_rate

TRUE = {
    "kappa": 0.5, "hH": 2.0, "hD": 8.0,
    "segments": {  # name: (baseline share, true beta)
        "OO_PI_LVR80": (0.12, 0.060), "OO_PI_LVR90": (0.08, 0.075),
        "INV_PI": (0.10, 0.050), "INV_IO": (0.15, 0.040),
        "OO_FIX2": (0.06, 0.090), "OO_IO": (0.09, 0.055),
    },
}
NOISE_SD = 0.05
LENDERS = [f"L{j}" for j in range(1, 9)]


def _market(rng, periods: int):
    """Common RBA-like steps, and each lender's (and our, index -1) delayed pass-through of them."""
    steps = [(t, float(rng.choice([-25, 25]))) for t in sorted(rng.choice(
        np.arange(6, periods - 12), size=3, replace=False))]
    passed = np.zeros((periods, len(LENDERS) + 1))
    for t, size in steps:
        for j, d in enumerate(rng.integers(0, 4, len(LENDERS) + 1)):
            passed[t + d:, j] += size
    return passed


def simulate(out_dir: Path, seed: int = 0, competitors: bool = False, reactive: bool = False,
             periods: int = 104) -> dict:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-07-07", periods=periods, freq="W-SUN")
    season = 0.03 * np.sin(np.arange(periods) * 2 * np.pi / 52)  # market-wide shock, hits all segs
    lamH, lamD = (1 - 2.0 ** (-1 / TRUE[h]) for h in ("hH", "hD"))
    passed = _market(rng, periods) if competitors else np.zeros((periods, len(LENDERS) + 1))
    rows, events, comp = [], [], []
    for seg, (s0, beta) in TRUE["segments"].items():
        r_star = np.zeros(periods)
        if competitors:
            names = list(rng.choice(LENDERS, size=rng.integers(6, 9), replace=False))
            cols = [LENDERS.index(n) for n in names]
            R = 620 + rng.normal(0, 20) + rng.normal(0, 15, len(names)) + passed[:, cols]
            for j in range(len(names)):  # sparse own moves, reversed later (specials); some
                for t in rng.integers(-26, periods, size=rng.integers(1, 4)):  # already running
                    R[max(t, 0):max(t + rng.integers(8, 27), 0), j] += (
                        rng.choice([-1, 1]) * rng.integers(5, 21))
            adj = rng.uniform(0, 10, len(names)).round(1)
            r_star = softmin_rate(R - adj, 1 / beta)
            comp += [{"period": dates[t].date(), "segment": seg, "lender": n,
                      "rate_pct": round(R[t, j] / 100, 4), "adj_bps": adj[j]}
                     for t in range(periods) for j, n in enumerate(names)]
        H, D = passed[:, -1].copy(), np.zeros(periods)
        clean_at = set(rng.choice(np.arange(8, periods - 10), size=3, replace=False).tolist())
        drift = 0.05 if reactive else 0.01  # natural position random walk step
        alpha = np.log(s0 / (1 - s0)) + np.cumsum(rng.normal(0, drift, periods))
        eps = rng.normal(0, NOISE_SD, periods)
        y, pH, pD, pR, last = np.zeros(periods), 0.0, 0.0, 0.0, -99
        for t in range(periods):
            move = None
            if t in clean_at:
                if rng.random() < 0.6:
                    move = (float(rng.choice([-10, -5, 5, 10])), 0.0, "funding cost", True)
                else:
                    move = (0.0, float(rng.choice([-15, -10, 10, 15])), "funding cost", True)
            elif (reactive and t >= 13 and t - last >= 8
                  and y[t - 1] - y[t - 13:t - 1].mean() < -1.05 * NOISE_SD):
                move = (-10.0, 0.0, "volume shortfall", False)
            if move:
                H[t:] += move[0]
                D[t:] += move[1]
                last = t
                events.append({"segment": seg, "period": dates[t].date(), "d_headline_bps": move[0],
                               "d_discretionary_bps": move[1], "reason": move[2], "clean": move[3]})
            # lag() written out step by step, because a reactive cut depends on last period's y
            pH, pD = pH + lamH * (H[t] - pH), pD + lamD * (D[t] - pD)
            pR = pR + lamH * (r_star[t] - r_star[0] - pR)
            y[t] = alpha[t] + season[t] - beta * (pH + TRUE["kappa"] * pD - pR) + eps[t]
        share = 1 / (1 + np.exp(-y))
        mkt = rng.poisson(4000, periods)
        us = np.maximum(1, np.round(share * mkt)).astype(int)
        apps = np.round(us * rng.normal(1.05, 0.02, periods)).astype(int)
        for i in range(periods):
            rows.append({"segment": seg, "period": dates[i].date(), "lodg_mkt": mkt[i],
                         "lodg_us": us[i], "apps": apps[i], "headline_cum_bps": H[i],
                         "discretionary_cum_bps": D[i]})
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "panel.csv", index=False)
    pd.DataFrame(events).sort_values(["segment", "period"]).to_csv(out_dir / "events.csv",
                                                                     index=False)
    if competitors:
        pd.DataFrame(comp).to_csv(out_dir / "competitors.csv", index=False)
    return TRUE
