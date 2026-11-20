"""Command line entry point:  uv run elasticity
{simulate,check-moves,estimate,forecast,export-excel,charts,validate,sensitivity}"""

import argparse
import dataclasses
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from mortgage_elasticity import panel, runlog
from mortgage_elasticity.config import Config
from mortgage_elasticity.runlog import log


def _load(args, cfg: Config) -> panel.Prepared:
    """Read and validate the input files; build the benchmark at the starting temperature."""
    from mortgage_elasticity.bayes import prior_T

    df = panel.load_panel(Path(args.panel), cfg.date_format, cfg.covariates)
    segs = list(dict.fromkeys(df["segment"]))
    ev = panel.load_events(Path(args.events), segs, cfg.date_format)
    comp, T = None, None
    if args.competitors:
        comp = panel.load_competitors(Path(args.competitors), cfg.exclude_lenders,
                                      df["period"].unique(), cfg.date_format)
        s0 = df.groupby("segment", sort=False)[["lodg_us", "lodg_mkt"]].sum()
        T = prior_T((s0["lodg_us"] / s0["lodg_mkt"]).to_numpy(), segs, cfg)
    prep = panel.prepare(df, ev, comp, T, cfg)
    log(f"Data: {len(prep.segs)} segments x {len(prep.periods)} weeks "
        f"({prep.periods[0].date()} to {prep.periods[-1].date()}), {int(prep.obs.sum())} "
        f"segment-weeks in the likelihood; {len(prep.moves('price'))} own price moves "
        f"({int(prep.moves('price')['all'].sum())} bank-wide), {len(prep.moves('other'))} logged "
        f"non-price changes; competitors: {'none' if comp is None else comp['lender'].nunique()}")
    return prep


def _start(out: Path, args, cfg: Config) -> None:
    out.mkdir(parents=True, exist_ok=True)
    runlog.start()
    log(f"Run at {datetime.now().isoformat(timespec='seconds')}: "
        f"{ {k: v for k, v in vars(args).items() if k != 'func'} }")
    log("Config: " + ", ".join(f"{k}={v}" for k, v in dataclasses.asdict(cfg).items()))


def _fit(args):
    from mortgage_elasticity.bayes import Fit
    return Fit.load(args.fit)


def cmd_simulate(args):
    from mortgage_elasticity.simulate import simulate
    truth = simulate(Path(args.out), seed=args.seed, u=args.u, reactive=args.reactive)
    print(f"Wrote synthetic panel.csv, events.csv, competitors.csv, truth.json to {args.out}: "
          f"{len(truth['segments'])} segments, {truth['n_price_events']} own price moves, true "
          f"uplift {truth['u']:.2%}/bp at the pooled share")


def cmd_check_moves(args):
    from mortgage_elasticity import diagnostics
    cfg, out = Config(), Path(args.out)
    _start(out, args, cfg)
    table = diagnostics.raw_move_check(_load(args, cfg), cfg, out / "check_moves")
    table.to_csv(out / "check_moves.csv", index=False)
    if len(table):
        with pd.option_context("display.float_format", "{:.2f}".format, "display.width", 160):
            log(table[table["segment"] == "TOTAL"].drop(columns=["segment", "post_weeks"])
                .to_string(index=False))
    log(f"Wrote {out / 'check_moves.csv'} and one table and plot per move in {out / 'check_moves'}")
    runlog.write(out / "check_moves_log.txt")


def cmd_estimate(args):
    from mortgage_elasticity import bayes, diagnostics
    cfg, out = Config(), Path(args.out)
    _start(out, args, cfg)
    prep = _load(args, cfg)
    diagnostics.raw_move_check(prep, cfg, out / "check_moves").to_csv(
        out / "check_moves.csv", index=False)

    fit = bayes.estimate(prep, cfg)
    fit.save(out / "fit.pkl")
    _write_outputs(fit, out)
    runlog.write(out / "run_log.txt")


def _write_outputs(fit, out: Path) -> None:
    from mortgage_elasticity import diagnostics, forecast, report
    prep, cfg = fit.prep, fit.cfg
    if fit.T_path is not None:
        fit.T_path.to_csv(out / "T_path.csv", index=False)
        log("Soft-min temperature by iteration (bp):\n"
            + fit.T_path.set_index("segment").round(1).to_string())
    lo, mid, hi = 100 * np.quantile(
        np.exp(fit.prior_log_u[:, prep.segs.index(_mid_segment(prep))]), cfg.q)
    log(f"Prior check: {cfg.interval:.0%} prior interval of u for {_mid_segment(prep)} is "
        f"{lo:.1f} to {hi:.1f}%/bp (median {mid:.1f})")

    segments = report.segments_table(fit)
    segments.to_csv(out / "segments.csv", index=False)
    report.global_table(fit).to_csv(out / "global.csv", index=False)
    natural = report.natural_share_table(fit)
    natural.to_csv(out / "natural_share.csv", index=False)
    report.natural_share_plots(fit, natural, out / "natural_share")
    accuracy = forecast.accuracy_table(fit)
    accuracy.to_csv(out / "accuracy.csv", index=False)
    diagnostics.estimators(fit).to_csv(out / "diagnostics_estimators.csv", index=False)
    ppc = diagnostics.ppc(fit)
    ppc.to_csv(out / "diagnostics_ppc.csv", index=False)

    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 160):
        log(segments[["segment", "lodg_us_per_week", "share_now_p50",
                      *[c for c in segments if c.startswith("u_p")], "T_bps", "n_price_moves",
                      "prior_share", "thin_volume", "sampler_warning"]].to_string(index=False))
        log(report.global_table(fit).to_string(index=False))
    alone = accuracy[accuracy["kind"] == "segment"]
    for move, g in alone.groupby("move_bps"):
        log(f"Accuracy, {move:+.0f}bp: {int(g['meets_5pct_price_effect'].sum())} of {len(g)} "
            f"segments within +/-{cfg.target_apps_rel:.0%} applications (price effect only), "
            f"{int(g['meets_5pct_full_predictive'].sum())} with drift and sampling noise")
    log(f"Posterior predictive checks: observed inside the simulated {cfg.interval:.0%} range in "
        f"{ppc['inside'].mean():.0%} of segment statistics")
    if segments["thin_volume"].any():
        log(f"Thin segments (< {cfg.thin_lodgements:.0f} own lodgements a week): "
            f"{segments.loc[segments['thin_volume'], 'segment'].tolist()}: read them in groups")
    log(f"Wrote segments.csv, global.csv, natural_share.csv (+ plots), accuracy.csv, "
        f"diagnostics_*.csv, check_moves.csv, fit.pkl and run_log.txt to {out}")


def _mid_segment(prep: panel.Prepared) -> str:
    """A mid-range segment for the prior check: the middle LVR band if present."""
    mid = prep.dims.index[prep.dims["lvr_band"] == "70-80"]
    return mid[0] if len(mid) else prep.segs[len(prep.segs) // 2]


def cmd_forecast(args):
    from mortgage_elasticity.forecast import forecast
    fit = _fit(args)
    segments = "ALL" if args.segments.upper() == "ALL" else [
        s.strip() for s in args.segments.split(";") if s.strip()]
    changes = {}
    for item in args.competitor or []:
        name, _, bps = item.partition("=")
        seg, _, lender = name.rpartition(":")
        changes[(seg, lender) if seg else lender] = float(bps)
    f = forecast(fit, segments, args.headline, args.discretionary, changes or None, args.horizon)
    lo, hi = (f"p{round(100 * q)}" for q in (fit.cfg.q[0], fit.cfg.q[2]))
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        labels = {
            "price_effect": "PRICE-EFFECT interval (natural share held; no competitor response, "
                            "no market movement)",
            "full_predictive": "FULL PREDICTIVE interval (natural share drifts on, counts are "
                               "noisy)"}
        for version, label in labels.items():
            print(f"\n{label}, {fit.cfg.interval:.0%} range ({lo} to {hi}):")
            g = f.summary[f.summary["version"] == version]
            print(g[["segment", "dx_bps", *[c for c in g if c.startswith(("multiplier", "share_pp",
                                                                          "apps_"))]]]
                  .to_string(index=False))
        g = f.summary[f.summary["version"] == "price_effect"]
        print("\nLong-run effect once the move has fully phased in (not the answer for the "
              "forecast horizon):")
        print(g[["segment", *[c for c in g if c.startswith("long_run")]]].to_string(index=False))
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        f.summary.to_csv(out / "forecast.csv", index=False)
        f.weekly.to_csv(out / "forecast_weekly.csv", index=False)
        print(f"\nWrote forecast.csv and forecast_weekly.csv to {out}")


def cmd_export_excel(args):
    from mortgage_elasticity.excel import export_excel
    print(f"Wrote {export_excel(_fit(args), Path(args.out))}")


def cmd_charts(args):
    from mortgage_elasticity import visuals
    written = visuals.gallery(_fit(args), Path(args.out), args.segment, args.move, args.margin,
                              outputs=Path(args.fit).parent, truth=args.truth)
    print(f"Wrote {len(written)} charts to {args.out}")


def cmd_validate(args):
    from mortgage_elasticity import validate
    fit, out = _fit(args), Path(args.out)
    _start(out, args, fit.cfg)
    table = getattr(validate, args.what)(fit)
    table.to_csv(out / f"validate_{args.what}.csv", index=False)
    log(f"Wrote {out / f'validate_{args.what}.csv'}")
    runlog.write(out / f"validate_{args.what}_log.txt")


def cmd_sensitivity(args):
    from mortgage_elasticity import validate
    fit, out = _fit(args), Path(args.out)
    _start(out, args, fit.cfg)
    table = validate.sensitivity(fit)
    table.to_csv(out / "sensitivity.csv", index=False)
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 160):
        log(table[table["kind"] == "all"].drop(columns=["kind", "name"]).to_string(index=False))
    log(f"Wrote {out / 'sensitivity.csv'}")
    runlog.write(out / "sensitivity_log.txt")


def _inputs(p):
    p.add_argument("--panel", required=True)
    p.add_argument("--events", required=True)
    p.add_argument("--competitors", help="competitor rates; without them the gap is our own move")
    p.add_argument("--out", default="outputs")


def _from_fit(p, out="outputs"):
    p.add_argument("--fit", default="outputs/fit.pkl", help="fit.pkl written by `estimate`")
    p.add_argument("--out", default=out)


def main():
    p = argparse.ArgumentParser(prog="elasticity")
    sub = p.add_subparsers(required=True)
    s = sub.add_parser("simulate", help="write a synthetic panel, event log and competitor file")
    s.add_argument("--out", default="data/sample")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--u", type=float, default=0.03, help="true uplift per bp")
    s.add_argument("--reactive", action="store_true",
                   help="also cut price where share has slipped")
    s.set_defaults(func=cmd_simulate)
    s = sub.add_parser("check-moves", help="raw-move check: no model; run this first")
    _inputs(s)
    s.set_defaults(func=cmd_check_moves)
    s = sub.add_parser("estimate", help="fit the model and write every output")
    _inputs(s)
    s.set_defaults(func=cmd_estimate)
    s = sub.add_parser("forecast", help="effect of a price change over the next weeks")
    _from_fit(s, out=None)
    s.add_argument("--segments", required=True, help="ALL, or segment names separated by ;")
    s.add_argument("--headline", type=float, required=True, help="headline change, bp (+ = dearer)")
    s.add_argument("--discretionary", type=float, default=0.0,
                   help="change in average discount given, bp (+ = less discount)")
    s.add_argument("--competitor", action="append", metavar="[SEGMENT:]LENDER=BPS",
                   help="a competitor's rate change in bp; repeatable")
    s.add_argument("--horizon", type=int)
    s.set_defaults(func=cmd_forecast)
    s = sub.add_parser("export-excel", help="write the workbook with live formulas")
    _from_fit(s, out="outputs/elasticity.xlsx")
    s.set_defaults(func=cmd_export_excel)
    s = sub.add_parser("charts", help="write the explanatory charts as PNGs")
    _from_fit(s, out="outputs/charts")
    s.add_argument("--segment", help="segment for the single-segment charts (default: largest)")
    s.add_argument("--move", type=float, default=-10.0, help="example price change, bp")
    s.add_argument("--margin", type=float,
                   help="margin on new lending, bp: adds the volume-against-margin illustration")
    s.add_argument("--truth", help="truth.json of a simulated panel: adds the true sensitivities")
    s.set_defaults(func=cmd_charts)
    s = sub.add_parser("validate", help="backtests: refits the model several times")
    s.add_argument("what", choices=["events", "quiet", "placebo"])
    _from_fit(s)
    s.set_defaults(func=cmd_validate)
    s = sub.add_parser("sensitivity", help="refit under other priors and settings")
    _from_fit(s)
    s.set_defaults(func=cmd_sensitivity)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
