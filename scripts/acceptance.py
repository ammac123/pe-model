"""Acceptance runs on simulated data (brief section 11). Slow: each run is a full estimate.

    uv run python scripts/acceptance.py [--seeds 10] [--workers 3] [--out outputs/acceptance]

For true uplift 3%, 8% and 1.5%/bp, for 3%/bp with reactive repricing switched on, and for 3%/bp
with segments that depart from the additive purpose / repayment / LVR pattern (with and without
the segment residual):
  coverage   share of segment-runs whose 80% interval for u_k contains the truth (want 70-90%)
  mu_u       |posterior median of exp(mu_u) / truth - 1|, averaged over runs (want < 25%)
  bias       median over segments of (u_p50 / truth - 1), averaged over runs (reactive: |.| < 20%)
Results are reported as they come out. Priors are not to be changed to make them pass.
"""

import argparse
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

# (label, true uplift, reactive repricing, segments exactly on the additive pattern, residual on)
CASES = [("u=3%", 0.03, False, True, False), ("u=8%", 0.08, False, True, False),
         ("u=1.5%", 0.015, False, True, False), ("u=3%, reactive", 0.03, True, True, False),
         ("u=3%, off-pattern", 0.03, False, False, False),
         ("u=3%, off-pattern, residual on", 0.03, False, False, True)]


def run(case: str, u: float, reactive: bool, on_pattern: bool, residual: bool, seed: int) -> dict:
    from mortgage_elasticity import bayes, panel
    from mortgage_elasticity.config import Config
    from mortgage_elasticity.simulate import simulate

    warnings.simplefilter("ignore")
    cfg = replace(Config(), segment_residual=residual)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)
        truth = simulate(path, seed=seed, u=u, reactive=reactive, pooled_share_truth=on_pattern)
        df = panel.load_panel(path / "panel.csv")
        segs = list(dict.fromkeys(df["segment"]))
        comp = panel.load_competitors(path / "competitors.csv", (), df["period"].unique())
        s0 = df.groupby("segment", sort=False)[["lodg_us", "lodg_mkt"]].sum()
        T = bayes.prior_T((s0["lodg_us"] / s0["lodg_mkt"]).to_numpy(), segs, cfg)
        prep = panel.prepare(df, panel.load_events(path / "events.csv", segs), comp, T, cfg)
    start = time.time()
    fit = bayes.estimate(prep, cfg)
    true_u = np.array([truth["segments"][s]["u"] for s in segs])
    lo, mid, hi = fit.q("u")
    return {"case": case, "seed": seed, "seconds": time.time() - start,
            "n_price_events": truth["n_price_events"], "additive_gap": truth["additive_gap"],
            "coverage": float(((lo <= true_u) & (true_u <= hi)).mean()),
            "mu_u_ratio": float(np.exp(np.median(fit.draws["mu_u"])) / truth["u"]),
            "median_bias": float(np.median(mid / true_u - 1)),
            "T_iterations": fit.T_path.shape[1] - 2, "T_converged": fit.T_converged,
            "max_rhat": fit.diag["max_rhat"], "min_ess": fit.diag["min_ess"],
            "divergences": fit.diag["divergences"], "reliable": fit.diag["reliable"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="outputs/acceptance")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(*case, seed) for case in CASES for seed in range(args.seeds)]
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn")) as pool:
        rows = []
        for row in pool.map(run, *zip(*jobs)):
            rows.append(row)
            pd.DataFrame(rows).to_csv(out / "runs.csv", index=False)
            print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()},
                  flush=True)
    runs = pd.DataFrame(rows)
    summary = runs.groupby("case", sort=False).agg(
        runs=("seed", "size"), coverage=("coverage", "mean"),
        mu_u_abs_error=("mu_u_ratio", lambda r: (r - 1).abs().mean()),
        median_bias=("median_bias", "mean"), T_converged=("T_converged", "mean"),
        reliable=("reliable", "mean"), max_seconds=("seconds", "max")).reset_index()
    summary["coverage_70_90"] = summary["coverage"].between(0.70, 0.90)
    summary["mu_u_within_25pct"] = summary["mu_u_abs_error"] < 0.25
    summary["bias_within_20pct"] = summary["median_bias"].abs() < 0.20
    summary["under_30_min"] = summary["max_seconds"] < 1800
    summary.to_csv(out / "summary.csv", index=False)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
