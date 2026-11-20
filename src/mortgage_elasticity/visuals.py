"""Charts that explain the model and its forecasts in business terms.

Every function returns a matplotlib Figure and saves nothing; `gallery` writes the whole set as
PNGs (`uv run elasticity charts`). notebooks/model_explorer.ipynb walks through them in order.

  how the model works    response_curve, phase_in, benchmark_weights
  what it learned        sensitivity_by_segment, sensitivity_grid, prior_vs_posterior, share_history
  scenarios, 1-D         forecast_path, scenario_comparison, move_size_curve, margin_tradeoff
  scenarios, 2-D / 3-D   segment_move_heatmap, competitor_response, phase_in_surface
  how far to trust it    raw_moves, backtest, accuracy_heatmap, uncertainty_vs_volume,
                         sensitivity_runs

Conventions: a negative price change is a cut. Volume effects are shown as the % change in
applications against "no price change" (multiplier - 1). Ranges are the fit's reporting interval
(80% by default). Unless a chart says otherwise the range is the price effect only: natural share
held where it is today, no competitor response.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from mortgage_elasticity import forecast, panel, plots
from mortgage_elasticity.bayes import Fit
from mortgage_elasticity.model import beta_from_uplift, multiplier, softmin_rate, step_response
from mortgage_elasticity.plots import AQUA, BLUE, GRID, INK, MUTED, ORANGE, SURFACE, filename

RED, MIDPOINT = "#e34948", "#f0efec"
SEQUENTIAL = LinearSegmentedColormap.from_list("sequential", ["#cde2fb", "#3987e5"])
DIVERGING = LinearSegmentedColormap.from_list("diverging", [RED, MIDPOINT, BLUE])  # loss - gain


def _signed(v: float, unit: str = "") -> str:
    return f"{v:+,.0f}{unit}" if abs(v) >= 0.5 else f"0{unit}"


PCT = FuncFormatter(lambda v, _: _signed(v, "%"))
PRICE = "our price change, bp (negative = cut)"
APPS = "change in applications against no price change"


# --- shared pieces ---------------------------------------------------------------------------

def _style(ax, title: str = "", xlabel: str = "", ylabel: str = "", note: str = "",
           grid: str = "y") -> None:
    """Title in ink, a one-line reading note under it, hairline grid, no frame."""
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold",
                 pad=10 + 12 * len(note.splitlines()))
    if note:
        ax.annotate(note, (0, 1), xycoords="axes fraction", xytext=(0, 6),
                    textcoords="offset points", color=MUTED, fontsize=8.5, va="bottom")
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def _axes(title: str, xlabel: str = "", ylabel: str = "", note: str = "", size=(9, 4.4),
          grid: str = "y"):
    fig, ax = plt.subplots(figsize=size, facecolor=SURFACE)
    _style(ax, title, xlabel, ylabel, note, grid)
    return fig, ax


def _legend(ax, ncol: int = 4, inches: float = 0.6, handles=None) -> None:
    """Legend under the axes, `inches` below its bottom edge whatever the chart's height."""
    drop = inches / (ax.figure.get_figheight() * ax.get_position().height)
    ax.legend(**({"handles": handles} if handles else {}), frameon=False, fontsize=8, labelcolor=MUTED, loc="upper left",
              bbox_to_anchor=(0, -drop), ncol=ncol, handlelength=1.6, columnspacing=1.4)


def _zero(ax, axis: str = "h") -> None:
    (ax.axhline if axis == "h" else ax.axvline)(0, color=MUTED, linewidth=0.8, zorder=1)


def _heat(ax, Z, xlabels, ylabels, cmap, norm, text) -> None:
    """Grid of cells with a surface-coloured gap between them and the value written in each."""
    Z = np.ma.masked_invalid(np.asarray(Z, float))
    ax.pcolormesh(Z, cmap=cmap, norm=norm, edgecolors=SURFACE, linewidth=2)
    for (i, j), v in np.ndenumerate(Z.filled(np.nan)):
        if np.isfinite(v):
            ax.text(j + 0.5, i + 0.5, text(i, j, v), ha="center", va="center", color=INK,
                    fontsize=7.5)
    ax.set_xticks(np.arange(len(xlabels)) + 0.5, xlabels)
    ax.set_yticks(np.arange(len(ylabels)) + 0.5, ylabels)
    ax.invert_yaxis()
    ax.spines["bottom"].set_visible(False)


def _surface(X, Y, Z, title: str, xlabel: str, ylabel: str, elev: float, azim: float) -> Figure:
    """3-D surface of a % change in applications, coloured red (loss) to blue (gain)."""
    fig = plt.figure(figsize=(8.5, 6.2), facecolor=SURFACE)
    with plt.rc_context({"grid.color": GRID, "axes.edgecolor": GRID}):
        ax = fig.add_subplot(projection="3d", facecolor=SURFACE)
    v = float(np.abs(Z).max())
    surf = ax.plot_surface(X, Y, Z, cmap=DIVERGING, norm=Normalize(-v, v), edgecolor=SURFACE,
                           linewidth=0.3, rstride=1, cstride=1, alpha=0.95)
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.set_zlabel(APPS, color=MUTED, fontsize=9)
    ax.zaxis.set_major_formatter(PCT)
    ax.tick_params(colors=MUTED, labelsize=8)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color(SURFACE)
    ax.view_init(elev=elev, azim=azim)
    bar = fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1, format=PCT)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=MUTED, labelsize=8, length=0)
    return fig


def _expit(z):
    return 1 / (1 + np.exp(-z))


def _interval(fit: Fit) -> str:
    return f"{fit.cfg.interval:.0%} range"


def _ordered(fit: Fit) -> list[int]:
    """Segment indices by purpose, repayment and LVR band, for charts that list segments."""
    d = fit.prep.dims
    key = [(panel.PURPOSES.index(p), panel.REPAYMENTS.index(r), fit.cfg.lvr_bands.index(b))
           for p, r, b in zip(d["purpose"], d["repayment"], d["lvr_band"])]
    return sorted(range(len(key)), key=key.__getitem__)


def price_effect(fit: Fit, segment: str, dx, horizon: int | None = None, draws: int = 1000):
    """Weekly share of one segment under each change dx in the price gap (bp), natural share
    held. Returns (baseline: draws x weeks, scenario: draws x len(dx) x weeks, long-run
    multiplier: draws x len(dx)). This is forecast.paths on an evenly thinned set of draws,
    for many dx at once; applications scale with share, so ratios of these are multipliers."""
    k, d = fit.prep.segs.index(segment), fit.draws
    i = np.linspace(0, fit.n - 1, min(draws, fit.n)).astype(int)
    dx = np.atleast_1d(np.asarray(dx, float))
    beta, level = d["beta"][i, k][:, None], d["level"][i, -1, k].astype(float)[:, None]
    v_T, x_T = fit.prep.gap_raw[-1, k], fit.x()[i, -1, k].astype(float)[:, None]
    felt = 1 - 2.0 ** (-np.arange(1, (horizon or fit.cfg.horizon) + 1) / d["h"][i][:, None])
    base = _expit(level - beta * (x_T + (v_T - x_T) * felt))
    target = (v_T + dx - x_T)[:, :, None]  # draws x dx x 1
    scn = _expit(level[:, :, None] - beta[:, :, None] * (x_T[:, :, None] + target * felt[:, None]))
    long_run = _expit(level - beta * (v_T + dx)) / _expit(level - beta * v_T)
    return base, scn, long_run


def _change(fit: Fit, segment: str, dx, **kw) -> np.ndarray:
    """% change in applications over the horizon, draws x len(dx)."""
    base, scn, _ = price_effect(fit, segment, dx, **kw)
    return 100 * (scn.sum(-1) / base.sum(-1)[:, None] - 1)


def _moves(lo: float = -30, hi: float = 30, step: float = 1) -> np.ndarray:
    return np.arange(lo, hi + step / 2, step)


# --- how the model works (formulas only, no fit needed) ----------------------------------------

def response_curve(s0: float = 0.10, uplifts=(0.012, 0.03, 0.074), moves=None) -> Figure:
    """Fully felt response to a price change at several sensitivities (uplift per bp), for a
    lender with share s0. The defaults are the low, middle and high of the starting range."""
    moves = _moves() if moves is None else np.asarray(moves, float)
    fig, ax = _axes("How applications respond to price, at three sensitivities", PRICE, APPS,
                    f"Once a move is fully felt, for a lender with {s0:.0%} of the market. "
                    "Sensitivity = % more applications per 1bp cut.")
    _zero(ax)
    for u, colour in zip(uplifts, (BLUE, ORANGE, AQUA)):
        y = 100 * (multiplier(moves, s0, beta_from_uplift(u, s0)) - 1)
        ax.plot(moves, y, color=colour, linewidth=2, label=f"{u:.1%} per bp")
        ax.annotate(f"{u:.1%} per bp", (moves[0], y[0]), xytext=(4, 0),
                    textcoords="offset points", color=MUTED, fontsize=8, va="bottom")
    ax.yaxis.set_major_formatter(PCT)
    _legend(ax)
    return fig


def phase_in(half_lives=(1.0, 2.0, 4.0), horizon: int = 12) -> Figure:
    """Share of a price move felt by each week after it, for several half-lives."""
    weeks = np.arange(horizon + 1)
    fig, ax = _axes("A price move takes weeks to be felt", "weeks since the move",
                    "share of the move felt, %",
                    "Half-life = weeks until half the move is felt. The legend gives the average "
                    f"felt over the first {horizon} weeks.")
    for h, colour in zip(half_lives, (BLUE, ORANGE, AQUA)):
        mean = step_response(weeks[1:], h).mean()
        ax.plot(weeks, 100 * step_response(weeks, h), color=colour, linewidth=2, marker="o",
                markersize=4, label=f"half-life {h:g} weeks: {mean:.0%} felt on average")
    ax.set_ylim(0, 105)
    _legend(ax, ncol=1)
    return fig


def benchmark_weights(fit: Fit, segment: str) -> Figure:
    """Each competitor's weight in the market benchmark r* for one segment, last week of data."""
    prep = fit.prep
    if prep.comp is None:
        raise ValueError("benchmark_weights: the model was fitted without competitors")
    snap = panel.benchmark_snapshot(prep.comp, prep.T[[segment]], prep.periods,
                                    fit.cfg.comp_ffill_periods).sort_values("eff_bps")
    fig, ax = _axes(f"{segment}: which competitors set the market benchmark", "weight in the "
                    "benchmark, %", "",
                    f"Benchmark r* = {snap['r_star_pct'].iloc[0]:.2f}%; cheapest competitor "
                    f"{snap['eff_bps'].min() / 100:.2f}%. Rates are after manual adjustments.",
                    size=(9, 0.42 * len(snap) + 1.6), grid="x")
    labels = [f"{ln}   {e / 100:.2f}%" for ln, e in zip(snap["lender"], snap["eff_bps"])]
    bars = ax.barh(labels, 100 * snap["weight"], height=0.6, color=BLUE)
    ax.bar_label(bars, fmt="%.0f%%", padding=4, color=MUTED, fontsize=8)
    ax.invert_yaxis()
    return fig


# --- what the model learned --------------------------------------------------------------------

def sensitivity_by_segment(fit: Fit, truth: dict | None = None) -> Figure:
    """Uplift per bp for every segment with its range, against the starting range (prior).
    truth: truth.json of a simulated panel; adds the true value of each segment."""
    prep, order = fit.prep, _ordered(fit)
    lo, mid, hi = 100 * fit.q("u")[:, order]
    names = [prep.segs[k] for k in order]
    fig, ax = _axes("Price sensitivity by segment", "uplift per bp: % more applications from a "
                    "1bp cut", "", f"Dot = middle estimate, bar = {_interval(fit)}. The grey band "
                    "is where the model started before seeing data.",
                    size=(9, 0.3 * len(names) + 1.8), grid="x")
    if fit.prior_log_u is not None:
        p_lo, _, p_hi = 100 * np.quantile(np.exp(fit.prior_log_u), fit.cfg.q)
        ax.axvspan(p_lo, p_hi, color=GRID, alpha=0.6, linewidth=0,
                   label=f"starting range, {fit.cfg.interval:.0%}")
    y = np.arange(len(names))
    for purpose, colour, label in zip(panel.PURPOSES, (BLUE, ORANGE),
                                      ("owner-occupier", "investor")):
        m = (prep.dims["purpose"].to_numpy()[order] == purpose)
        ax.hlines(y[m], lo[m], hi[m], color=colour, linewidth=2.5, alpha=0.45)
        ax.plot(mid[m], y[m], "o", color=colour, markersize=6, label=label)
    if truth:
        ax.plot([100 * truth["segments"][s]["u"] for s in names], y, "|", color=INK,
                markersize=11, markeredgewidth=1.6, label="true value (simulated data)")
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    _legend(ax)
    return fig


def sensitivity_grid(fit: Fit) -> Figure:
    """Middle estimate of uplift per bp as a grid: purpose and repayment by LVR band."""
    d = fit.prep.dims.assign(u=100 * fit.q("u")[1])
    d["row"] = d["purpose"] + " " + d["repayment"]
    rows = [f"{p} {r}" for p in panel.PURPOSES for r in panel.REPAYMENTS if f"{p} {r}" in set(d["row"])]
    bands = [b for b in fit.cfg.lvr_bands if b in set(d["lvr_band"])]
    Z = d.pivot(index="row", columns="lvr_band", values="u").reindex(index=rows, columns=bands)
    fig, ax = _axes("Where customers are most price-sensitive", "LVR band", "",
                    "Middle estimate of uplift per bp. Darker = more sensitive.",
                    size=(7.5, 0.55 * len(rows) + 1.6), grid="")
    _heat(ax, Z.to_numpy(), bands, rows, SEQUENTIAL, Normalize(np.nanmin(Z), np.nanmax(Z)),
          lambda i, j, v: f"{v:.1f}%")
    return fig


def prior_vs_posterior(fit: Fit, reference: float | None = 0.08) -> Figure:
    """The typical uplift per bp before the data (starting range) and after. reference: a rule
    of thumb to mark, as a fraction per bp."""
    cfg = fit.cfg
    post = 100 * np.exp(fit.draws["mu_u"])
    x = np.linspace(0.05, max(10.0, post.max() * 1.1), 600)
    prior = np.exp(-0.5 * ((np.log(x / 100) - np.log(cfg.prior_u_median)) / cfg.prior_u_log_sd) ** 2) / x
    dens, edges = np.histogram(post, bins=30)
    fig, ax = _axes("What the data added to the starting judgment", "typical uplift per bp, %",
                    "relative plausibility",
                    "Each curve is scaled to the same height. A narrow curve means the figure is "
                    "known tightly.")
    ax.fill_between(x, prior / prior.max(), color=MUTED, alpha=0.18, linewidth=0)
    ax.plot(x, prior / prior.max(), color=MUTED, linewidth=2, label="starting range (prior)")
    centres = (edges[:-1] + edges[1:]) / 2
    ax.fill_between(centres, dens / dens.max(), color=BLUE, alpha=0.25, linewidth=0)
    ax.plot(centres, dens / dens.max(), color=BLUE, linewidth=2, label="after the data (posterior)")
    lo, mid, hi = np.quantile(post, cfg.q)
    ax.annotate(f"{mid:.1f}% ({lo:.1f} to {hi:.1f})", (mid, 1.0), xytext=(8, -2),
                textcoords="offset points", color=INK, fontsize=9)
    if reference:
        ax.axvline(100 * reference, color=ORANGE, linewidth=1.5,
                   label=f"rule of thumb, {reference:.0%} per bp")
    ax.set_yticks([])
    ax.set_xlim(0, x[-1])
    ax.set_ylim(0, 1.12)
    _legend(ax)
    return fig


def share_history(fit: Fit, segment: str) -> Figure:
    """Top: actual, fitted and natural share. Bottom: the price gap behind the difference, as
    set (raw) and as customers have felt it (phased in). Logged events are marked on both."""
    prep, k, q = fit.prep, fit.prep.segs.index(segment), fit.cfg.q
    level, x = fit.draws["level"][:, :, k].astype(float), fit.x()[:, :, k].astype(float)
    natural = 100 * np.quantile(_expit(level), q, axis=0)
    fitted = 100 * np.median(_expit(level - fit.draws["beta"][:, [k]] * x), axis=0)
    actual = 100 * prep.L[:, k] / np.where(prep.M[:, k] > 0, prep.M[:, k], np.nan)
    ev = prep.events[[segment in s for s in prep.events["segments"]]]
    fig, (top, low) = plt.subplots(2, 1, figsize=(9, 6.6), facecolor=SURFACE, sharex=True,
                                   height_ratios=(3, 2), gridspec_kw={"hspace": 0.28})
    _style(top, f"{segment}: share, and how much of it is price", "",
           "share of market lodgements, %",
           "Natural share = the share we would have had with the price gap left where it "
           "started. Fitted minus natural = the effect of price.")
    plots._mark_events(top, ev)
    top.fill_between(prep.periods, natural[0], natural[2], color=ORANGE, alpha=0.18, linewidth=0,
                     label=f"natural share, {_interval(fit)}")
    top.plot(prep.periods, natural[1], color=ORANGE, linewidth=2, label="natural share")
    top.plot(prep.periods, fitted, color=BLUE, linewidth=2, label="fitted share")
    top.scatter(prep.periods, actual, s=9, color=INK, alpha=0.45, linewidths=0, zorder=3,
                label="actual share")
    top.set_ylim(bottom=0)
    _style(low, "", "", "price gap against the market, bp")
    low.set_title("Price gap: above zero we are dearer than at the start, below zero cheaper",
                  loc="left", color=MUTED, fontsize=8.5, pad=18)
    plots._mark_events(low, ev)
    _zero(low)
    low.plot(prep.periods, prep.gap_raw[:, k], color=AQUA, linewidth=1.5, drawstyle="steps-post",
             label="gap as set")
    low.plot(prep.periods, np.median(x, axis=0), color=BLUE, linewidth=2,
             label="gap as felt by customers")
    _legend(low, inches=0.35, handles=[*top.get_legend_handles_labels()[0],
                                       *low.get_legend_handles_labels()[0][-2:]])
    return fig


# --- scenarios: one dimension -----------------------------------------------------------------

def _cols(table: pd.DataFrame, prefix: str) -> list[str]:
    """The low, middle and high columns of a forecast table for one measure."""
    return [c for c in table.columns if c.startswith(f"{prefix}_p")]


def forecast_path(fc: forecast.Forecast, segment: str | None = None, measure: str = "apps",
                  title: str | None = None) -> Figure:
    """Week-by-week path of one forecast with both ranges. measure: "apps" (% change in
    applications) or "share" (share of market, %). segment: default the last row (TOTAL if the
    forecast covers several segments)."""
    segment = segment or fc.summary["segment"].iloc[-1]
    s = fc.summary[fc.summary["segment"] == segment].set_index("version")
    w = fc.weekly[fc.weekly["segment"] == segment]
    prefix = "multiplier" if measure == "apps" else "share_pp"
    cols = _cols(w, prefix)
    scale = (lambda v: 100 * (v - 1)) if measure == "apps" else (lambda v: v)
    rng = f"{int(cols[2].rsplit('p', 1)[1]) - int(cols[0].rsplit('p', 1)[1])}% range"
    dx = s["dx_bps"].iloc[0]
    fig, ax = _axes(title or f"{segment}: week by week after a {dx:+.0f}bp change in the price gap",
                    "weeks after the change",
                    APPS if measure == "apps" else "share of market lodgements, %",
                    "Dark band: price effect only. Light band: what to expect in practice, with "
                    "market drift and weekly noise.")
    for version, alpha, label in [("full_predictive", 0.15, f"with drift and noise, {rng}"),
                                  ("price_effect", 0.35, f"price effect only, {rng}")]:
        g = w[w["version"] == version]
        ax.fill_between(g["week"], scale(g[cols[0]]), scale(g[cols[2]]), color=BLUE, alpha=alpha,
                        linewidth=0, label=label)
    g = w[w["version"] == "price_effect"]
    ax.plot(g["week"], scale(g[cols[1]]), color=BLUE, linewidth=2, marker="o", markersize=4,
            label="middle estimate")
    weeks = g["week"].to_numpy()
    if measure == "apps":
        _zero(ax)
        ax.yaxis.set_major_formatter(PCT)
        avg = scale(s.loc["price_effect", cols[1]])
        long_run = scale(s.loc["price_effect", "long_run_multiplier_p50"])
        ax.hlines(long_run, weeks[0], weeks[-1], color=ORANGE, linewidth=1.5,
                  label=f"long-run effect once fully felt ({long_run:+.0f}%)")
        ax.hlines(avg, weeks[0], weeks[-1], color=AQUA, linewidth=1.5,
                  label=f"average over the {len(weeks)} weeks ({avg:+.0f}%)")
    else:
        ax.axhline(s["baseline_share_pp_p50"].iloc[0], color=MUTED, linewidth=1,
                   label="share with no change")
    ax.set_xticks(weeks)
    _legend(ax, ncol=3)
    return fig


def scenario_comparison(scenarios: dict[str, forecast.Forecast], segment: str | None = None,
                        title: str = "Scenarios compared: applications over the forecast horizon"
                        ) -> Figure:
    """One row per named scenario: middle estimate, price-effect range (thick) and full
    predictive range (thin) of applications over the horizon, against no change."""
    fig, ax = _axes(title, "applications over the horizon", "",
                    "Dot = middle estimate. Thick bar: price effect only. Thin bar: with market "
                    "drift and weekly noise.", size=(9, 0.5 * len(scenarios) + 1.8), grid="x")
    for y, (name, fc) in enumerate(scenarios.items()):
        seg = segment or fc.summary["segment"].iloc[-1]
        s = fc.summary[fc.summary["segment"] == seg].set_index("version")
        lo, mid, hi = _cols(s, "apps")
        first = y == 0
        ax.hlines(y, s.loc["full_predictive", lo], s.loc["full_predictive", hi], color=BLUE,
                  linewidth=1.5, alpha=0.6, label="with drift and noise" if first else None)
        ax.hlines(y, s.loc["price_effect", lo], s.loc["price_effect", hi], color=BLUE,
                  linewidth=7, alpha=0.4, label="price effect only" if first else None)
        m, base = s.loc["price_effect", mid], s.loc["price_effect", "baseline_apps_p50"]
        ax.plot(m, y, "o", color=BLUE, markersize=7, markeredgecolor=SURFACE, markeredgewidth=2,
                label="middle estimate" if first else None)
        ax.annotate(f"{m:,.0f}  ({_signed(100 * (m / base - 1), '%')})",
                    (s.loc["full_predictive", hi], y),
                    xytext=(8, 0), textcoords="offset points", color=MUTED, fontsize=8,
                    va="center")
        if first:
            ax.axvline(base, color=MUTED, linewidth=1, label=f"no change ({base:,.0f})")
    ax.set_yticks(range(len(scenarios)), list(scenarios))
    ax.set_ylim(len(scenarios) - 0.5, -0.5)
    ax.margins(x=0.16)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    _legend(ax)
    return fig


def move_size_curve(fit: Fit, segment: str, moves=None, draws: int = 1000) -> Figure:
    """Change in applications over the horizon against the size of the price move, with its
    range, and the larger long-run effect."""
    moves = _moves() if moves is None else np.asarray(moves, float)
    base, scn, long_run = price_effect(fit, segment, moves, draws=draws)
    lo, mid, hi = np.quantile(100 * (scn.sum(-1) / base.sum(-1)[:, None] - 1), fit.cfg.q, axis=0)
    weeks = base.shape[1]
    fig, ax = _axes(f"{segment}: what each size of price move does to applications", PRICE, APPS,
                    f"Blue: average over the next {weeks} weeks, while the move is still sinking "
                    "in. Orange: once fully felt.")
    _zero(ax)
    ax.fill_between(moves, lo, hi, color=BLUE, alpha=0.2, linewidth=0,
                    label=f"next {weeks} weeks, {_interval(fit)}")
    ax.plot(moves, mid, color=BLUE, linewidth=2, label=f"next {weeks} weeks")
    ax.plot(moves, 100 * (np.median(long_run, axis=0) - 1), color=ORANGE, linewidth=1.5,
            label="long run, once fully felt")
    for dx in (-10.0, 10.0):
        if moves.min() <= dx <= moves.max():
            v = float(np.interp(dx, moves, mid))
            ax.plot(dx, v, "o", color=BLUE, markersize=7, markeredgecolor=SURFACE,
                    markeredgewidth=2)
            ax.annotate(f"{dx:+.0f}bp: {v:+.0f}%", (dx, v), xytext=(8, 6),
                        textcoords="offset points", color=INK, fontsize=9)
    ax.yaxis.set_major_formatter(PCT)
    _legend(ax)
    return fig


def margin_tradeoff(fit: Fit, segment: str, margin_bps: float, moves=None,
                    draws: int = 1000) -> Figure:
    """Illustration, not a model output: volume against income on new lending, where income is
    volume x (margin + price change). margin_bps is your own figure for the margin earned on new
    lending today. It ignores any cost of the same change on existing customers."""
    moves = _moves() if moves is None else np.asarray(moves, float)
    vol = 1 + _change(fit, segment, moves, draws=draws) / 100
    income = 100 * (vol * (margin_bps + moves) / margin_bps - 1)
    lo, mid, hi = np.quantile(income, fit.cfg.q, axis=0)
    best = moves[int(np.argmax(mid))]
    fig, ax = _axes(f"{segment}: volume against margin on new lending", PRICE,
                    "change against no price change",
                    f"Illustration: income = applications x margin, margin today {margin_bps:.0f}bp"
                    f", next {fit.cfg.horizon} weeks, new lending only.")
    _zero(ax)
    ax.plot(moves, 100 * (np.median(vol, axis=0) - 1), color=BLUE, linewidth=2,
            label="applications")
    ax.fill_between(moves, lo, hi, color=ORANGE, alpha=0.2, linewidth=0,
                    label=f"income on new lending, {_interval(fit)}")
    ax.plot(moves, mid, color=ORANGE, linewidth=2, label="income on new lending")
    ax.plot(best, mid.max(), "o", color=ORANGE, markersize=7, markeredgecolor=SURFACE,
            markeredgewidth=2)
    edge = best in (moves.min(), moves.max())
    ax.annotate(f"income highest {'at the edge of this range,' if edge else 'near'} {best:+.0f}bp "
                f"({mid.max():+.0f}%)", (best, mid.max()), xytext=(0, 10),
                textcoords="offset points", color=INK, fontsize=9,
                ha="left" if best == moves.min() else "right" if edge else "center")
    ax.yaxis.set_major_formatter(PCT)
    _legend(ax)
    return fig


# --- scenarios: two and three dimensions -------------------------------------------------------

def segment_move_heatmap(fit: Fit, moves=(-25, -15, -10, -5, 5, 10, 15, 25),
                         measure: str = "pct", draws: int = 500) -> Figure:
    """Every segment against every move size: middle estimate of the change in applications
    over the horizon. measure: "pct" (% change) or "apps" (number of applications)."""
    order = _ordered(fit)
    names = [fit.prep.segs[k] for k in order]
    M, conv = forecast.trailing(fit)
    Z = np.empty((len(names), len(moves)))
    for i, (k, seg) in enumerate(zip(order, names)):
        base, scn, _ = price_effect(fit, seg, moves, draws=draws)
        Z[i] = np.median(100 * (scn.sum(-1) / base.sum(-1)[:, None] - 1), axis=0)
        if measure == "apps":
            Z[i] *= np.median(base.sum(-1)) * M[k] * conv[k] / 100
    unit = "%" if measure == "pct" else ""
    fig, ax = _axes("Where a price move changes volume most", PRICE, "",
                    ("% change in" if measure == "pct" else "Extra") + " applications over the "
                    f"next {fit.cfg.horizon} weeks, middle estimate. Blue = gain, red = loss.",
                    size=(1.0 * len(moves) + 2.2, 0.32 * len(names) + 1.6), grid="")
    v = np.abs(Z).max()
    _heat(ax, Z, [_signed(m) for m in moves], names, DIVERGING, Normalize(-v, v),
          lambda i, j, z: _signed(z, unit))
    return fig


def _competitor_grid(fit: Fit, segment: str, who: str, ours, theirs, draws: int):
    """% change in applications (theirs x ours, middle estimate) when the lenders `who` move."""
    prep = fit.prep
    if prep.comp is None:
        raise ValueError("competitor_response: the model was fitted without competitors")
    snap = panel.benchmark_snapshot(prep.comp, prep.T[[segment]], prep.periods,
                                    fit.cfg.comp_ffill_periods)
    q, T = snap["eff_bps"].to_numpy(), prep.T[segment]
    if who == "all":
        hit, label = np.ones(len(q), bool), "every competitor"
    else:
        name = snap["lender"].iloc[int(np.argmin(q))] if who == "cheapest" else who
        if name not in set(snap["lender"]):
            raise ValueError(f"competitor_response: {name} is not in the benchmark of {segment}")
        hit = (snap["lender"] == name).to_numpy()
        label = f"{name} ({'the cheapest, ' if who == 'cheapest' else ''}weight "
        label += f"{snap['weight'].to_numpy()[hit][0]:.0%} in the benchmark)"
    shift = np.array([softmin_rate(q + hit * d, T)[0] for d in theirs]) - softmin_rate(q, T)[0]
    gap = np.asarray(ours, float)[None, :] - shift[:, None]  # theirs x ours
    Z = np.median(_change(fit, segment, gap.ravel(), draws=draws), axis=0).reshape(gap.shape)
    return Z, label


def competitor_response(fit: Fit, segment: str, who: str = "cheapest", ours=None, theirs=None,
                        view: str = "heatmap", elev: float = 24, azim: float = -60,
                        draws: int = 300) -> Figure:
    """Our move against a competitor's move: middle estimate of the change in applications over
    the horizon. who: "cheapest", "all", or a lender name. view: "heatmap" or "3d"."""
    step = 5 if view == "heatmap" else 2.5
    ours = _moves(-25, 25, step) if ours is None else np.asarray(ours, float)
    theirs = _moves(-25, 25, step) if theirs is None else np.asarray(theirs, float)
    Z, label = _competitor_grid(fit, segment, who, ours, theirs, draws)
    title = f"{segment}: our move against a competitor's move"
    ylabel = "competitor's price change, bp"
    if view == "3d":
        fig = _surface(*np.meshgrid(ours, theirs), Z, f"{title}\n", PRICE, ylabel, elev, azim)
        fig.axes[0].text2D(0, 0.98, f"Competitor moving: {label}.",
                           transform=fig.axes[0].transAxes, color=MUTED, fontsize=8.5)
        return fig
    fig, ax = _axes(title, PRICE, ylabel,
                    f"Competitor moving: {label}. % change in applications over the next "
                    f"{fit.cfg.horizon} weeks.",
                    size=(0.72 * len(ours) + 1.8, 0.4 * len(theirs) + 1.8), grid="")
    v = np.abs(Z).max()
    _heat(ax, Z, [_signed(m) for m in ours], [_signed(m) for m in theirs], DIVERGING,
          Normalize(-v, v), lambda i, j, z: _signed(z, "%"))
    return fig


def phase_in_surface(fit: Fit, segment: str, moves=None, elev: float = 24, azim: float = -60,
                     draws: int = 300) -> Figure:
    """3-D: size of the move x weeks since the move -> change in that week's applications
    (middle estimate). Shows the effect building up as the move is felt."""
    moves = _moves(-30, 30, 2.5) if moves is None else np.asarray(moves, float)
    base, scn, _ = price_effect(fit, segment, moves, draws=draws)
    Z = np.median(100 * (scn / base[:, None, :] - 1), axis=0).T  # weeks x moves
    weeks = np.arange(1, base.shape[1] + 1)
    return _surface(*np.meshgrid(moves, weeks), Z,
                    f"{segment}: how the effect of a move builds week by week", PRICE,
                    "weeks after the move", elev, azim)


# --- how far to trust it -----------------------------------------------------------------------

def raw_moves(check: pd.DataFrame) -> Figure:
    """No model: what applications did after each past price move (check_moves.csv, TOTAL rows),
    beside what each assumed uplift says they should have done."""
    t = check[check["segment"] == "TOTAL"].reset_index(drop=True)
    implied = [c for c in t.columns if c.startswith("implied_ratio_")]
    fig, ax = _axes("Past price moves: what happened against what each sensitivity implies",
                    "change in weekly applications, after against before the move", "",
                    "No model involved. If the dots sit near one set of ticks, that sensitivity "
                    "fits the raw history.", size=(9, 0.42 * len(t) + 1.9), grid="x")
    _zero(ax, "v")
    y = np.arange(len(t))
    vals = 100 * (t[["apps_ratio", *implied]].to_numpy() - 1)
    ax.hlines(y, vals.min(1), vals.max(1), color=GRID, linewidth=1.5)
    for c, colour in zip(implied, (ORANGE, AQUA)):
        ax.plot(100 * (t[c] - 1), y, "|", color=colour, markersize=13, markeredgewidth=2.5,
                label=f"implied by {c.split('u=')[1].replace('/bp', ' per bp')}")
    ax.plot(100 * (t["apps_ratio"] - 1), y, "o", color=INK, markersize=6, label="actual")
    ax.set_yticks(y, [f"{d}  {what}  ({g:+.0f}bp)" for d, what, g in
                      zip(t["date"], t["description"], t["d_gap_bps"])])
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PCT)
    _legend(ax)
    return fig


def backtest(table: pd.DataFrame, segment: str = "TOTAL") -> Figure:
    """A backtest table (validate_events.csv or validate_quiet.csv): the predicted range against
    what happened, each as a % of the middle prediction."""
    t = table[table["segment"] == segment].reset_index(drop=True)
    labels = ([f"{d}  {what}" for d, what in zip(t["date"], t["description"])]
              if "description" in t else [f"{d}  no price move" for d in t["origin"]])
    mid = t["predicted_p50"]
    fig, ax = _axes(f"Backtest: forecast range against what happened ({segment})",
                    "actual and predicted range, against the middle prediction", "",
                    "Each forecast was made with the following weeks hidden from the model. "
                    f"{int(t['covered'].sum())} of {len(t)} actuals fall inside the range.",
                    size=(9, 0.42 * len(t) + 1.9), grid="x")
    y = np.arange(len(t))
    _zero(ax, "v")
    ax.hlines(y, 100 * (t["predicted_low"] / mid - 1), 100 * (t["predicted_high"] / mid - 1),
              color=BLUE, linewidth=7, alpha=0.4, label="predicted range")
    ax.plot(100 * (t["actual"] / mid - 1), y, "D", color=INK, markersize=5, label="actual")
    for yi, inside, hi in zip(y, t["covered"], 100 * (t["predicted_high"] / mid - 1)):
        ax.annotate("inside" if inside else "outside", (ax.get_xlim()[1], yi), xytext=(4, 0),
                    textcoords="offset points", color=INK if not inside else MUTED, fontsize=8,
                    va="center", annotation_clip=False)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PCT)
    _legend(ax)
    return fig


def accuracy_heatmap(accuracy: pd.DataFrame, kind: str = "groups",
                     version: str = "full_predictive", target: float = 0.05) -> Figure:
    """How wide the forecast range of applications is (accuracy table), by move size, for
    kind = "groups" or "segment". A tick marks cells inside the +/-target."""
    a = accuracy[(accuracy["kind"] == "segment") == (kind == "segment")]
    names = list(dict.fromkeys(a["name"]))
    Z = 100 * a.pivot(index="name", columns="move_bps",
                      values=f"apps_halfwidth_rel_{version}").reindex(names)
    what = ("price effect only" if version == "price_effect"
            else "with market drift and weekly noise")
    fig, ax = _axes("How precise the forecast is, by move size", PRICE, "",
                    f"Half-width of the range of applications, {what}.\n"
                    f"✓ = inside the ±{target:.0%} target. Lighter = more precise.",
                    size=(1.0 * Z.shape[1] + 2.4, 0.34 * len(names) + 1.8), grid="")
    _heat(ax, Z.to_numpy(), [_signed(m) for m in Z.columns], names, SEQUENTIAL,
          Normalize(np.nanmin(Z.to_numpy()), np.nanmax(Z.to_numpy())),
          lambda i, j, v: f"±{v:.1f}%" + (" ✓" if v <= 100 * target else ""))
    return fig


def uncertainty_vs_volume(fit: Fit, accuracy: pd.DataFrame, move: float = -10) -> Figure:
    """Forecast precision of each segment against its weekly volume, for one move size: why thin
    segments should be read in groups."""
    a = accuracy[(accuracy["kind"] == "segment") & (accuracy["move_bps"] == move)]
    vol = pd.Series(fit.prep.L.mean(0), index=fit.prep.segs)[a["name"]].to_numpy()
    cfg = fit.cfg
    fig, ax = _axes(f"Small segments forecast less precisely ({move:+.0f}bp move)",
                    "our lodgements a week (log scale)",
                    "half-width of the range of applications, %",
                    "One pair of dots per segment. The gap between blue and orange is what price "
                    "cannot control: market drift and weekly noise.")
    for version, colour, label in [("price_effect", BLUE, "price effect only"),
                                   ("full_predictive", ORANGE, "with drift and noise")]:
        ax.plot(vol, 100 * a[f"apps_halfwidth_rel_{version}"], "o", color=colour, markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
    ax.axhline(100 * cfg.target_apps_rel, color=MUTED, linewidth=1,
               label=f"target, ±{cfg.target_apps_rel:.0%}")
    ax.axvline(cfg.thin_lodgements, color=MUTED, linewidth=1, linestyle=(0, (3, 3)),
               label=f"thin segment, below {cfg.thin_lodgements:.0f} a week")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylim(bottom=0)
    _legend(ax)
    return fig


def sensitivity_runs(table: pd.DataFrame, name: str = "ALL") -> Figure:
    """The typical uplift per bp when the model is refitted under other judgments
    (sensitivity.csv). The base run is in blue."""
    t = table[table["name"] == name].reset_index(drop=True)
    fig, ax = _axes(f"Does the answer depend on our judgment calls? ({name})",
                    "uplift per bp, %", "",
                    "Each row refits the model with one setting changed. Rows that sit on the "
                    "base run show the setting does not matter.",
                    size=(9, 0.4 * len(t) + 1.8), grid="x")
    y, base = np.arange(len(t)), t["run"] == "base"
    ax.axvline(100 * t.loc[base, "u_p50"].iloc[0], color=BLUE, linewidth=1, alpha=0.5)
    for m, colour in [(~base, MUTED), (base, BLUE)]:
        ax.hlines(y[m], 100 * t.loc[m, "u_low"], 100 * t.loc[m, "u_high"], color=colour,
                  linewidth=2.5, alpha=0.45)
        ax.plot(100 * t.loc[m, "u_p50"], y[m], "o", color=colour, markersize=6)
    ax.set_yticks(y, t["run"])
    ax.invert_yaxis()
    ax.margins(x=0.25)
    return fig


# --- everything at once ------------------------------------------------------------------------

def gallery(fit: Fit, out_dir: Path, segment: str | None = None, move_bps: float = -10.0,
            margin_bps: float | None = None, outputs: Path | None = None,
            truth: Path | None = None) -> list[Path]:
    """Write every chart as a PNG into out_dir. segment: the segment the single-segment charts
    show (default: our largest). outputs: the folder with check_moves.csv, validate_*.csv and
    sensitivity.csv, for the charts that read them (skipped where a file is absent)."""
    prep = fit.prep
    segment = segment or prep.segs[int(prep.L.mean(0).argmax())]
    one = forecast.forecast(fit, segment, move_bps)
    accuracy = forecast.accuracy_table(fit)
    charts = {
        "response_curve": lambda: response_curve(float(prep.s0[prep.segs.index(segment)])),
        "phase_in": phase_in,
        "sensitivity_by_segment": lambda: sensitivity_by_segment(
            fit, json.loads(Path(truth).read_text()) if truth else None),
        "sensitivity_grid": lambda: sensitivity_grid(fit),
        "prior_vs_posterior": lambda: prior_vs_posterior(fit),
        "share_history": lambda: share_history(fit, segment),
        "forecast_path_apps": lambda: forecast_path(one),
        "forecast_path_share": lambda: forecast_path(one, measure="share"),
        "scenario_comparison": lambda: scenario_comparison(
            {f"{m:+.0f}bp": forecast.forecast(fit, segment, m) for m in (-25, -10, -5, 5, 10, 25)},
            title=f"{segment}: applications over the next {fit.cfg.horizon} weeks, by move"),
        "move_size_curve": lambda: move_size_curve(fit, segment),
        "segment_move_heatmap": lambda: segment_move_heatmap(fit),
        "segment_move_heatmap_apps": lambda: segment_move_heatmap(fit, measure="apps"),
        "phase_in_surface": lambda: phase_in_surface(fit, segment),
        "accuracy_groups": lambda: accuracy_heatmap(accuracy, target=fit.cfg.target_apps_rel),
        "accuracy_segments": lambda: accuracy_heatmap(accuracy, "segment",
                                                      target=fit.cfg.target_apps_rel),
        "uncertainty_vs_volume": lambda: uncertainty_vs_volume(fit, accuracy, move_bps),
    }
    if prep.comp is not None:
        charts |= {
            "benchmark_weights": lambda: benchmark_weights(fit, segment),
            "competitor_response": lambda: competitor_response(fit, segment),
            "competitor_response_3d": lambda: competitor_response(fit, segment, view="3d"),
        }
    if margin_bps:
        charts["margin_tradeoff"] = lambda: margin_tradeoff(fit, segment, margin_bps)
    for name, chart in [("check_moves", raw_moves), ("validate_events", backtest),
                        ("validate_quiet", backtest), ("sensitivity", sensitivity_runs)]:
        path = Path(outputs) / f"{name}.csv" if outputs else None
        if path is not None and path.exists():
            charts[name] = lambda path=path, chart=chart: chart(pd.read_csv(path))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, chart in charts.items():
        fig = chart()
        written.append(out_dir / f"{filename(name)}.png")
        fig.savefig(written[-1], dpi=130, bbox_inches="tight", facecolor=SURFACE)
        plt.close(fig)
    return written
