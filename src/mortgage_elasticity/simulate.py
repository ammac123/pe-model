"""Synthetic panel, event log and competitor file with known parameters, for testing end to end.

The data follow the model in bayes.py: 20 segments (purpose x repayment x LVR band, several thin),
6-8 competitors per segment, and beta-binomial lodgement counts whose share depends on a drifting
natural level and on the lagged gap x = lag(H + kappa D - (r* - r*_1)), r* = softmin(eff, 1/beta_k).

Own moves (about 10, all logged as type=price): RBA-like steps passed on by everyone (us included)
after each lender's own 0-3 week delay, bank-wide repricings, and moves in a group of segments.
Non-price changes (type=other) shift the natural level; one lands in the same week as a price move.
reactive=True: we also cut 10bp in a segment whose share has slipped (at most n_reactive_max times),
and the natural level drifts more, so shortfalls persist. Those cuts are logged as ordinary price
moves. Our repricing leans back towards the market, and move sizes shrink when the true uplift is
above 3%/bp: otherwise shares run away to most of the market, which no lender's data look like.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mortgage_elasticity.model import phi_from_half_life, softmin_rate
from mortgage_elasticity.panel import PURPOSES, REPAYMENTS

BANDS = ("<=60", "60-70", "70-80", "80-90", ">90")
LENDERS = [f"L{j}" for j in range(1, 9)]
TRUE = {
    "kappa": 0.5, "h": 2.0, "c": 800.0,
    "a_inv": -0.2, "a_io": 0.1,  # log-uplift offsets: investors a little less sensitive
    "c_lvr": (-0.15, -0.05, 0.0, 0.05, 0.15),  # ... and high-LVR borrowers a little more
    "sigma_g": 0.01, "sigma_v": 0.015, "sigma_v_reactive": 0.04,
    "mkt_per_week": {("OO", "PI"): 2500, ("OO", "IO"): 500, ("INV", "PI"): 1200, ("INV", "IO"): 800},
    "band_weight": (0.8, 1.0, 1.4, 0.6, 0.15),  # the >90 band is thin
}


def _segments(n_segments: int) -> pd.DataFrame:
    rows = [{"segment": f"{p}_{r}_{b}", "is_inv": p == "INV", "is_io": r == "IO", "band": i,
             "mkt": TRUE["mkt_per_week"][p, r] * TRUE["band_weight"][i]}
            for p in PURPOSES for r in REPAYMENTS for i, b in enumerate(BANDS)]
    return pd.DataFrame(rows).iloc[:n_segments].reset_index(drop=True)


def _market(rng, periods: int, n_rba: int):
    """Common RBA-like steps, and each lender's (and our, index -1) delayed pass-through of them."""
    when = sorted(rng.choice(np.arange(6, periods - 14), size=n_rba, replace=False))
    passed = np.zeros((periods, len(LENDERS) + 1))
    for t in when:
        size = float(rng.choice([-25, 25]))
        for j, d in enumerate(rng.integers(0, 4, len(LENDERS) + 1)):
            passed[t + d:, j] += size
    return passed, [int(t) for t in when]


def _spaced(rng, n: int, lo: int, hi: int, gap: int, taken=()) -> list[int]:
    """n weeks in [lo, hi), at least `gap` apart from each other and from `taken`."""
    out: list[int] = []
    for t in rng.permutation(np.arange(lo, hi)):
        if all(abs(int(t) - s) >= gap for s in [*out, *taken]):
            out.append(int(t))
        if len(out) == n:
            break
    return sorted(out)


def _group(rng, seg: pd.DataFrame) -> np.ndarray:
    """A plausible repricing group: a purpose, a repayment type, an LVR band, or one or two cells."""
    kind = rng.choice(["purpose", "repayment", "band", "cells"])
    if kind == "purpose":
        m = seg["is_inv"].to_numpy() == bool(rng.integers(2))
    elif kind == "repayment":
        m = seg["is_io"].to_numpy() == bool(rng.integers(2))
    elif kind == "band":
        m = seg["band"].to_numpy() == rng.choice(seg["band"].unique())
    else:
        m = np.isin(np.arange(len(seg)), rng.choice(len(seg), size=min(2, len(seg)), replace=False))
    if m.all() or not m.any():  # must leave some segments untouched
        m = np.arange(len(seg)) == rng.integers(len(seg))
    return m


def simulate(out_dir: Path, seed: int = 0, u: float = 0.03, reactive: bool = False,
             periods: int = 104, n_segments: int = 20, competitors: bool = True,
             n_bank_moves: int = 4, n_segment_moves: int = 6, n_rba: int = 2,
             n_other: int = 3, n_reactive_max: int = 6, volume_scale: float = 1.0,
             pooled_share_truth: bool = True) -> dict:
    seg = _segments(n_segments)
    K = len(seg)
    dates = pd.date_range("2024-07-07", periods=periods, freq="W-SUN")
    phi = phi_from_half_life(TRUE["h"])
    u_k = u * np.exp(TRUE["a_inv"] * (seg["is_inv"] - 0.5) + TRUE["a_io"] * (seg["is_io"] - 0.5)
                     + np.asarray(TRUE["c_lvr"])[seg["band"]]).to_numpy()
    # The model defines uplift at the pooled share over the panel, u_k = beta_k (1 - s0_k), and
    # the pooled share depends on beta. Repeat the simulation from the same seed, each pass
    # setting beta from the previous pass's pooled share, until the two agree. With
    # pooled_share_truth=False beta is set once from the starting share instead, so the true
    # log u_k depart a little from the additive purpose / repayment / LVR pattern.
    scale = min(1.0, 0.03 / u)  # a lender with very price-sensitive customers moves in small steps
    pooled = None
    for _ in range(12 if pooled_share_truth else 1):
        rng = np.random.default_rng(seed)
        s0 = rng.uniform(0.05, 0.14, K)
        beta = u_k / (1 - (s0 if pooled is None else pooled))

        # competitors and the benchmark
        passed, rba_at = (_market(rng, periods, n_rba) if competitors
                          else (np.zeros((periods, len(LENDERS) + 1)), []))
        r_star, comp = np.zeros((periods, K)), []
        lender_level = rng.normal(0, 15, len(LENDERS))
        for k, name in enumerate(seg["segment"] if competitors else []):
            cols = np.sort(rng.choice(len(LENDERS), size=rng.integers(6, 9), replace=False))
            R = 620 + rng.normal(0, 20) + lender_level[cols] + rng.normal(0, 5, len(cols)) \
                + passed[:, cols]
            for j in range(len(cols)):  # sparse own moves, reversed later (specials); some
                for t in rng.integers(-26, periods, size=rng.integers(1, 4)):  # already running
                    R[max(t, 0):max(t + rng.integers(8, 27), 0), j] += round(
                        scale * rng.choice([-1, 1]) * rng.integers(5, 21))
            adj = rng.uniform(0, 10, len(cols)).round(1)
            r_star[:, k] = softmin_rate(R - adj, 1 / beta[k])
            comp += [{"period": dates[t].date(), "segment": name, "lender": LENDERS[c],
                      "rate_pct": round(R[t, j] / 100, 4), "adj_bps": adj[j]}
                     for t in range(periods) for j, c in enumerate(cols)]

        # scheduled own moves and non-price changes: week -> list of (mask, dH, dD, type, description)
        plan: dict[int, list] = {t: [] for t in range(periods)}
        everyone = np.ones(K, bool)
        for t in rba_at:
            plan[t].append((everyone, 0.0, 0.0, "price", "RBA pass-through"))  # sizes are in `passed`
        move_at = _spaced(rng, n_bank_moves + n_segment_moves, 8, periods - 14, 5, rba_at)
        bank = set(rng.choice(move_at, size=min(n_bank_moves, len(move_at)), replace=False).tolist())
        cum = np.zeros(K)  # our cumulative own move: repricing leans back towards the market
        for t in move_at:
            mask = everyone if t in bank else _group(rng, seg)
            lean = -np.sign(cum[mask].mean())
            sign = lean if lean and rng.random() < 0.8 else float(rng.choice([-1, 1]))
            size = round(scale * sign * float(rng.choice([5, 10, 15])))
            dH, dD = (size, 0.0) if rng.random() < 0.7 else (0.0, size)
            cum[mask] += dH + TRUE["kappa"] * dD
            what = "bank-wide repricing" if t in bank else "segment repricing"
            plan[t].append((mask, dH, dD, "price",
                            f"{what} ({'headline' if dH else 'discretionary'})"))
        jumps = []
        other_at = _spaced(rng, max(n_other - 1, 0), 8, periods - 14, 5, [*rba_at, *move_at])
        if n_other and move_at:  # one non-price change lands in the same week as a price move
            other_at = sorted([*other_at, move_at[len(move_at) // 2]])
        for i, t in enumerate(other_at):
            mask = everyone if i % 3 == 0 else _group(rng, seg)
            size = float(rng.choice([-1, 1]) * rng.uniform(0.08, 0.25))
            jumps.append(size)
            plan[t].append((mask, size, 0.0, "other",
                            ["broker campaign", "credit policy change", "commission change"][i % 3]))

        # natural level
        sd_v = TRUE["sigma_v_reactive"] if reactive else TRUE["sigma_v"]
        g = np.cumsum(rng.normal(0, TRUE["sigma_g"], periods)) \
            + 0.03 * np.sin(np.arange(periods) * 2 * np.pi / 52)  # market-wide, hits all segments
        level = np.log(s0 / (1 - s0)) + g[:, None] + np.cumsum(rng.normal(0, sd_v, (periods, K)), 0)

        H, D = np.tile(passed[:, [-1]], (1, K)), np.zeros((periods, K))
        eta, x, events = np.zeros((periods, K)), np.zeros((periods, K)), []
        p, last, n_reactive = np.zeros(K), np.full(K, -99), np.zeros(K, int)
        for t in range(periods):
            for mask, a, b, kind, what in plan[t]:
                if kind == "price":
                    H[t:, mask] += a
                    D[t:, mask] += b
                    last[mask] = t
                else:
                    level[t:, mask] += a
                events.append((t, mask, kind, what))
            if reactive and t >= 13:  # cut where share has slipped and we haven't just repriced
                slipped = ((eta[t - 1] - eta[t - 13:t - 1].mean(0) < -0.08) & (t - last >= 8)
                           & (n_reactive < 1))
                for k in np.flatnonzero(slipped)[:max(n_reactive_max - n_reactive.sum(), 0)]:
                    H[t:, k] -= round(scale * 10.0)
                    last[k], n_reactive[k] = t, n_reactive[k] + 1
                    events.append((t, np.arange(K) == k, "price", "volume shortfall"))
            # lag() written out step by step, because a reactive cut depends on last week's share
            gap = H[t] + TRUE["kappa"] * D[t] - (r_star[t] - r_star[0])
            p = gap.copy() if t == 0 else p + phi * (gap - p)
            x[t], eta[t] = p, level[t] - beta * p
        pi = np.clip(1 / (1 + np.exp(-eta)), 1e-9, 1 - 1e-9)
        season = 1 + 0.1 * np.sin(np.arange(periods) * 2 * np.pi / 52 + 1.0)
        mkt = rng.poisson(volume_scale * seg["mkt"].to_numpy() * season[:, None])
        us = rng.binomial(mkt, rng.beta(pi * TRUE["c"], (1 - pi) * TRUE["c"]))
        apps = np.round(us * rng.normal(1.05, 0.02, us.shape)).astype(int)
        new = us.sum(0) / mkt.sum(0)
        pooled = new if pooled is None else 0.5 * (pooled + new)  # damped: stable at high u

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = seg["segment"].to_numpy()
    pd.DataFrame({
        "segment": np.tile(names, periods), "period": np.repeat(dates.date, K),
        "lodg_mkt": mkt.ravel(), "lodg_us": us.ravel(), "apps": apps.ravel(),
        "headline_cum_bps": H.ravel(), "discretionary_cum_bps": D.ravel(),
    }).sort_values(["segment", "period"], kind="stable").to_csv(out_dir / "panel.csv", index=False)
    pd.DataFrame([{"date": dates[t].date(), "segments": "ALL" if m.all() else ";".join(names[m]),
                   "type": kind, "description": what} for t, m, kind, what in events],
                 columns=["date", "segments", "type", "description"]
                 ).to_csv(out_dir / "events.csv", index=False)
    if competitors:
        pd.DataFrame(comp).to_csv(out_dir / "competitors.csv", index=False)
    # the truth, stated as the model defines it; "u" is the typical one (balanced design)
    u_true = beta * (1 - us.sum(0) / mkt.sum(0))
    truth = {"u": float(np.exp(np.log(u_true).mean())), "u_design": u, "h": TRUE["h"],
             "kappa": TRUE["kappa"], "c": TRUE["c"],
             "a_inv": TRUE["a_inv"], "a_io": TRUE["a_io"], "c_lvr": list(TRUE["c_lvr"]),
             "segments": {n: {"s0": float(s), "u": float(uk), "beta": float(b)}
                          for n, s, uk, b in zip(names, us.sum(0) / mkt.sum(0), u_true, beta)},
             "additive_gap": float(np.abs(np.log(u_true / u_k)).max()),  # 0 = exactly additive
             "jumps": jumps, "n_price_events": sum(e[2] == "price" for e in events),
             "reactive": reactive, "seed": seed}
    (out_dir / "truth.json").write_text(json.dumps(truth, indent=1))
    # the latent paths, periods x segments in panel.csv order (sorted by segment name)
    order = np.argsort(names, kind="stable")
    np.savez_compressed(out_dir / "truth_paths.npz", segments=names[order], level=level[:, order],
                        x=x[:, order], r_star=r_star[:, order])
    return truth
