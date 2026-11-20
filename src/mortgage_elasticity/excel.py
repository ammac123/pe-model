"""Excel export: a workbook with live formulas that reproduces the price-effect forecast.

  segments   share_now, beta (low / middle / high), T, half-life, conversion, prior_share
  benchmark  each competitor's last-week effective rate, its weight in r*, and an editable
             scenario rate; r* now and under the scenario
  scenario   enter our headline and discretionary change per segment; the sheet computes dx, the
             weekly multipliers 1 / (s + (1 - s) EXP(beta dx (1 - (1 - phi)^tau))), their average
             over the horizon at each beta, and the long-run multiplier

The sheet starts from a settled baseline (past moves fully phased in) with h at its median, and its
range reflects the uncertainty in sensitivity only.
"""

from pathlib import Path

import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter as col

from mortgage_elasticity import panel, report
from mortgage_elasticity.bayes import Fit

INPUT = PatternFill("solid", fgColor="FFF2CC")  # cells the user may edit
BOLD = Font(bold=True)


def _header(ws, row: int, names: list[str]) -> None:
    for j, name in enumerate(names, start=1):
        ws.cell(row, j, name).font = BOLD


def export_excel(fit: Fit, path: Path) -> Path:
    prep, cfg = fit.prep, fit.cfg
    lo, hi = f"p{round(100 * cfg.q[0])}", f"p{round(100 * cfg.q[2])}"
    seg = report.segments_table(fit)
    h = float(np.median(fit.draws["h"]))
    wb = Workbook()

    # segments: row 2 + k
    ws = wb.active
    ws.title = "segments"
    _header(ws, 1, ["segment", "share_now", f"beta_{lo}", "beta_p50", f"beta_{hi}", "T_bps",
                    "h_p50", "conversion", "prior_share"])
    for k, r in enumerate(seg.itertuples(), start=2):
        row = [r.segment, r.share_now_p50, getattr(r, f"beta_{lo}"), r.beta_p50,
               getattr(r, f"beta_{hi}"), r.T_bps, h, r.conversion, r.prior_share]
        for j, v in enumerate(row, start=1):
            ws.cell(k, j, None if isinstance(v, float) and np.isnan(v) else v)
    seg_row = {s: k for k, s in enumerate(seg["segment"], start=2)}

    # benchmark: one block of rows per segment
    block = {}
    if prep.comp is not None:
        wb_ = wb.create_sheet("benchmark")
        _header(wb_, 1, ["segment", "lender", "rate_pct", "adj_bps", "eff_bps", "weight",
                         "scenario_rate_pct", "eff_scenario_bps", "exp_now", "exp_scenario",
                         "r_star_now_bps", "r_star_scenario_bps"])
        wb_["G1"].fill = INPUT
        snap = panel.benchmark_snapshot(prep.comp, prep.T, prep.periods, cfg.comp_ffill_periods)
        r = 2
        for s, g in snap.groupby("segment", sort=False):
            a, b = r, r + len(g) - 1
            block[s] = a
            T = f"segments!$F${seg_row[s]}"
            for row in g.itertuples():
                wb_.cell(r, 1, s)
                wb_.cell(r, 2, row.lender)
                wb_.cell(r, 3, row.rate_pct)
                wb_.cell(r, 4, row.adj_bps)
                wb_.cell(r, 5, row.eff_bps)
                wb_.cell(r, 6, f"=I{r}/SUM($I${a}:$I${b})")
                wb_.cell(r, 7).fill = INPUT
                wb_.cell(r, 8, f'=IF(G{r}="",E{r},100*G{r}-D{r})')
                wb_.cell(r, 9, f"=EXP(-(E{r}-MIN($E${a}:$E${b}))/{T})")
                wb_.cell(r, 10, f"=EXP(-(H{r}-MIN($H${a}:$H${b}))/{T})")
                wb_.cell(r, 11, f"=MIN($E${a}:$E${b})-{T}*LN(SUM($I${a}:$I${b}))")
                wb_.cell(r, 12, f"=MIN($H${a}:$H${b})-{T}*LN(SUM($J${a}:$J${b}))")
                r += 1

    # scenario: row 3 + k; weekly multipliers in three blocks to the right
    ws = wb.create_sheet("scenario")
    n = cfg.horizon
    first = 16  # first weekly column (P)
    start = {q: first + i * n for i, q in enumerate((lo, "p50", hi))}
    ws.cell(1, 2, "inputs: change in our rate, bp (+ = dearer)").font = BOLD
    ws.cell(1, 9, f"{n}-week applications multiplier: {cfg.interval:.0%} range from sensitivity "
                  "only").font = BOLD
    ws.cell(1, 12, "long-run multiplier (not the answer for the next "
                   f"{n} weeks)").font = BOLD
    _header(ws, 2, ["segment", "d_headline_bps", "d_discretionary_bps", "kappa", "d_r_star_bps",
                    "dx_bps", "share_now", "phi", f"multiplier_{lo}", "multiplier_p50",
                    f"multiplier_{hi}", f"long_run_{lo}", "long_run_p50", f"long_run_{hi}", ""]
            + [f"week_{t}_beta_{q}" for q in (lo, "p50", hi) for t in range(1, n + 1)])
    for k, s in enumerate(seg["segment"], start=3):
        sr = seg_row[s]
        ws.cell(k, 1, s)
        for j in (2, 3):
            ws.cell(k, j, 0).fill = INPUT
        ws.cell(k, 4, prep.kappa)
        ws.cell(k, 5, f"=benchmark!L{block[s]}-benchmark!K{block[s]}" if s in block else 0)
        ws.cell(k, 6, f"=B{k}+D{k}*C{k}-E{k}")
        ws.cell(k, 7, f"=segments!B{sr}")
        ws.cell(k, 8, f"=1-2^(-1/segments!G{sr})")
        for i, (q, beta_col) in enumerate(zip((lo, "p50", hi), "CDE")):
            beta = f"segments!${beta_col}${sr}"
            for t in range(1, n + 1):
                ws.cell(k, start[q] + t - 1,
                        f"=1/($G{k}+(1-$G{k})*EXP({beta}*$F{k}*(1-(1-$H{k})^{t})))")
            a, b = col(start[q]), col(start[q] + n - 1)
            ws.cell(k, 9 + i, f"=AVERAGE({a}{k}:{b}{k})")
            ws.cell(k, 12 + i, f"=1/($G{k}+(1-$G{k})*EXP({beta}*$F{k}))")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return path
