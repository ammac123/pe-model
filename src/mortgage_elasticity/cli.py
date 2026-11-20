"""Command line entry point:  uv run elasticity {simulate,estimate}"""

import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from mortgage_elasticity import alpha, events, fe, panel, pool
from mortgage_elasticity.config import Priors, Settings
from mortgage_elasticity.model import u_from_beta


def cmd_simulate(args):
    from mortgage_elasticity.simulate import simulate
    truth = simulate(Path(args.out), competitors=args.competitors, reactive=args.reactive)
    files = "panel.csv, events.csv" + (", competitors.csv" if args.competitors else "")
    print(f"Wrote synthetic {files} to {args.out}")
    print("True parameters:", truth)


def _fit(df, ev, s0, beta0, pr, st, args, xkw):
    """Steps 4a-5a for one benchmark: timing grid, fixed effects, event studies, posteriors."""
    # 4a fixed effects + grid over kappa / half-lives
    prior = {"kappa": pr.kappa, "hH": pr.h_headline, "hD": pr.h_discretionary}
    if args.no_grid:
        grid, best = pd.DataFrame(), prior
    else:
        grid, best, gain = fe.grid_search(df, st.kappa_grid, st.hH_grid, st.hD_grid, prior,
                                          st.grid_min_gain, **xkw)
        if best == prior:
            print(f"Grid winner cuts SSR by only {gain:.1%} vs the priors "
                  f"(< grid_min_gain {st.grid_min_gain:.0%}): keeping the priors")
    print(f"Timing/shading used: {best}")
    dfx = panel.add_perceived_price(df, best["kappa"], best["hH"], best["hD"], **xkw)
    fe_betas = fe.betas_from_fit(fe.fit_twfe(dfx))

    # 4b event studies
    es, ev_summary = events.event_study(dfx, ev, st.event_pre, st.event_post, st.min_dx_bps,
                                        pr.prior_sd_frac * beta0)
    print(ev_summary)
    print(f"Usable events: {int(es['clean'].sum())} of {len(es)} estimable "
          f"({int((es['other_changes'] != '').sum())} have other changes; "
          f"{int((~es['flagged_clean'].astype(bool)).sum())} flagged clean=FALSE)")

    # 5a prior -> posterior, per segment and pooled across segments
    post = pool.posterior_betas(s0, es, pr.u, pr.prior_sd_frac)
    pooled = pool.pooled_betas(s0, es, pr.u, pr.prior_sd_frac, pr.segment_sd_frac)
    print(f"Shared scale on the u={pr.u:.0%} assumption: {pooled['shared_scale'].iloc[0]:.2f} "
          f"± {pooled['shared_scale_sd'].iloc[0]:.2f} (from {int(post['n_clean_events'].sum())} "
          "usable events across all segments)")
    return grid, best, dfx, fe_betas, es, post, pooled


def cmd_estimate(args):
    pr, st = Priors(), Settings()
    if not (pr.u > 0 and 0 < pr.segment_sd_frac < pr.prior_sd_frac):
        raise ValueError("config: need u > 0 and 0 < segment_sd_frac < prior_sd_frac")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1-2 panel + diagnostics
    df = panel.load_panel(Path(args.panel), st.date_format)
    ev = panel.load_events(Path(args.events), df["segment"].unique(), st.date_format)
    s0 = panel.baseline_share(df, st.s0_window)
    beta0 = pr.u / (1 - s0)
    conv = panel.conversion_report(df)
    print(f"Panel: {df.segment.nunique()} segments x {df.period.nunique()} periods, {len(ev)} events")

    # competitor benchmark (gap model); without it x is our own move, demeaned across segments
    comp, T = None, pd.Series(np.nan, index=s0.index)
    if args.competitors:
        comp = panel.load_competitors(Path(args.competitors), st.exclude_lenders,
                                      df["period"].unique(), st.date_format)
        T = 1 / beta0 if st.softmin_T_bps is None else pd.Series(st.softmin_T_bps, index=s0.index)
        df = panel.add_benchmark(df, comp, T, st.comp_ffill_periods, st.min_lenders)
        snap = panel.benchmark_snapshot(comp, T, df["period"].unique(), st.comp_ffill_periods)
        n_now = snap.groupby("segment").size()
        print(f"Benchmark: gap model, T={T.min():.1f}-{T.max():.1f}bp, "
              f"lenders/segment={n_now.min()}-{n_now.max()}")
    xkw = {"demean": st.demean_x_by_period}

    grid, best, dfx, fe_betas, es, post, pooled = _fit(df, ev, s0, beta0, pr, st, args, xkw)
    # MNL: T = 1/beta, so T set from the prior is only right if the prior is. Re-solve once at the
    # TWFE estimate: unlike beta_pool it doesn't lean on the (thin, mixed-motive) event log.
    if comp is not None and st.softmin_T_bps is None and st.resolve_T:
        ok = lambda b: np.isfinite(b) & (b > 0)
        b_fe = fe_betas.set_index("segment")["beta_fe"].reindex(T.index)
        b_pool = pooled.set_index("segment")["beta_pool"].reindex(T.index)
        T_new = (1 / b_fe).where(ok(b_fe), (1 / b_pool).where(ok(b_pool), T))
        if (~ok(b_fe)).any():
            warnings.warn("beta_fe not positive: T falls back to 1/beta_pool or 1/beta_prior for "
                          + ", ".join(map(str, T.index[~ok(b_fe)])))
        T = T_new
        df = panel.add_benchmark(df, comp, T, st.comp_ffill_periods, st.min_lenders)
        snap = panel.benchmark_snapshot(comp, T, df["period"].unique(), st.comp_ffill_periods)
        print(f"Re-solving T = 1/beta_fe: T={T.min():.1f}-{T.max():.1f}bp. The rest is the second "
              "pass, the one reported:")
        grid, best, dfx, fe_betas, es, post, pooled = _fit(df, ev, s0, beta0, pr, st, args, xkw)

    # Excel contract: never rename/remove/reorder these columns; new ones go on the end
    params = (s0.reset_index().merge(post, on="segment").merge(fe_betas, on="segment", how="left"))
    params["u_post"] = u_from_beta(params["beta_post"], params["s0"])
    params["u_fe"] = u_from_beta(params["beta_fe"], params["s0"])
    params["kappa"], params["h_headline"], params["h_discretionary"] = (
        best["kappa"], best["hH"], best["hD"])

    # 5b optional hierarchical (a panel model: it never sees the events or the usable flag)
    hier = None
    if args.bayes:
        from mortgage_elasticity.bayes import fit_hierarchical
        hier = fit_hierarchical(dfx, s0, pr.u, pr.prior_sd_frac)
        params = params.merge(hier[["segment", "mean", "sd"]].rename(
            columns={"mean": "beta_hier", "sd": "beta_hier_sd"}), on="segment", how="left")

    params = params.merge(pooled, on="segment", how="left")
    last = dfx.groupby("segment", sort=False).tail(1).set_index("segment")
    params["x_basis"] = "gap" if comp is not None else "move"
    params["softmin_T_bps"] = params["segment"].map(T)
    params["r_star_now_pct"] = (params["segment"].map(last["r_star_bps"]) / 100
                                if comp is not None else np.nan)
    params["n_lenders_now"] = params["segment"].map(n_now) if comp is not None else np.nan
    # Excel's multiplier treats s0 as the share at today's x: x should be ~flat over the s0 window
    tail_x = dfx.groupby("segment", sort=False).tail(st.s0_window).groupby("segment")["x"]
    params["x_drift_s0_window"] = params["segment"].map(tail_x.max() - tail_x.min())

    # 6 natural position
    track = alpha.track_alpha(dfx, params.set_index("segment")["beta_post"],
                              st.min_periods_alpha_mle, st.q_over_r)

    meta = pd.DataFrame({"key": [
        "run_at", "panel", "events", "competitors", "periods", "x_basis", "softmin_T_bps",
        "demean_x_by_period", "grid", "bayes",
    ], "value": [
        datetime.now().isoformat(timespec="seconds"), args.panel, args.events,
        args.competitors or "", f"{df.period.min().date()} to {df.period.max().date()}",
        params["x_basis"].iloc[0],
        "" if comp is None else (st.softmin_T_bps if st.softmin_T_bps is not None
                                 else "1/beta_fe per segment" if st.resolve_T
                                 else "1/beta_prior per segment"),
        st.demean_x_by_period and comp is None, not args.no_grid, args.bayes,
    ]})

    # 7 export: Excel reads the `parameters` sheet
    params.to_csv(out / "parameters.csv", index=False)
    with pd.ExcelWriter(out / "parameters.xlsx") as xw:
        params.to_excel(xw, sheet_name="parameters", index=False)
        es.to_excel(xw, sheet_name="event_estimates", index=False)
        if len(grid):
            grid.to_excel(xw, sheet_name="timing_grid", index=False)
        conv.to_excel(xw, sheet_name="conversion_check", index=False)
        track.to_excel(xw, sheet_name="natural_share", index=False)
        if hier is not None:
            hier.to_excel(xw, sheet_name="hierarchical", index=False)
        if comp is not None:
            snap.to_excel(xw, sheet_name="benchmark", index=False)
        meta.to_excel(xw, sheet_name="meta", index=False)

    cols = ["segment", "s0", "n_clean_events", "u_post", "beta_post", "beta_pool", "beta_fe",
            *(["beta_hier"] if hier is not None else []), "prior_weight", "own_weight"]
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 120):
        print(params[cols].to_string(index=False))
    for c in ["beta_post", "beta_pool", "beta_fe", *(["beta_hier"] if hier is not None else [])]:
        bad = params.loc[params[c] <= 0, "segment"].tolist()
        if bad:
            warnings.warn(f"{c} <= 0 for {bad}: don't use it in Excel")
    drift = params.loc[params["x_drift_s0_window"] > 5, "segment"].tolist()
    if drift:
        warnings.warn(f"x moved > 5bp within the s0 window for {drift}: s0 is not the share at "
                      "today's price")
    print(f"Wrote {out / 'parameters.xlsx'} and parameters.csv")


def main():
    warnings.formatwarning = lambda msg, *a, **k: f"WARNING: {msg}\n"
    p = argparse.ArgumentParser(prog="elasticity")
    sub = p.add_subparsers(required=True)
    s = sub.add_parser("simulate", help="write a synthetic panel + event log")
    s.add_argument("--out", default="data/sample")
    s.add_argument("--competitors", action="store_true", help="also write competitors.csv")
    s.add_argument("--reactive", action="store_true",
                   help="also cut price when volume falls short (logged clean=FALSE)")
    s.set_defaults(func=cmd_simulate)
    e = sub.add_parser("estimate", help="run the estimation pipeline")
    e.add_argument("--panel", required=True)
    e.add_argument("--events", required=True)
    e.add_argument("--competitors", help="competitor rates: switches x to the gap vs the market")
    e.add_argument("--out", default="outputs")
    e.add_argument("--no-grid", action="store_true", help="skip the kappa/half-life grid; use priors")
    e.add_argument("--bayes", action="store_true", help="also fit the PyMC hierarchical model")
    e.set_defaults(func=cmd_estimate)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
