"""Static charts (PNG). One y-axis per chart; colour marks identity, text stays in ink colours."""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # first three categorical slots


def filename(name: str) -> str:
    """A segment or event name as a safe file name (<=60 -> le60, >90 -> gt90)."""
    name = str(name).replace("<=", "le").replace(">=", "ge").replace("<", "lt").replace(">", "gt")
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")


def _axes(title: str, ylabel: str, size=(9, 4.2)):
    fig, ax = plt.subplots(figsize=size, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    return fig, ax


def _mark_events(ax, events) -> None:
    """Vertical rules: solid = own price move, dashed = logged non-price change."""
    seen = set()
    for ev in events.itertuples():
        price = ev.type == "price"
        label = "own price move" if price else "logged non-price change"
        ax.axvline(ev.period, color=MUTED, linewidth=0.9, linestyle="-" if price else (0, (3, 3)),
                   alpha=0.6, zorder=1, label=None if label in seen else label)
        seen.add(label)


def _finish(fig, ax, path: Path) -> None:
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED, loc="upper left",
              bbox_to_anchor=(0, -0.1), ncol=6, handlelength=1.6, columnspacing=1.4)
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def natural_share(path: Path, seg: str, periods, share_obs, fitted, natural, events,
                  interval: float) -> None:
    """natural: (low, middle, high) x periods. Shares are fractions; plotted in percent."""
    fig, ax = _axes(f"{seg}: actual, fitted and natural share", "share of market lodgements, %")
    _mark_events(ax, events)
    ax.fill_between(periods, 100 * natural[0], 100 * natural[2], color=ORANGE, alpha=0.18,
                    linewidth=0, label=f"natural share, {interval:.0%} range")
    ax.plot(periods, 100 * natural[1], color=ORANGE, linewidth=2, label="natural share")
    ax.plot(periods, 100 * fitted, color=BLUE, linewidth=2, label="fitted share")
    ax.scatter(periods, 100 * np.asarray(share_obs), s=9, color=INK, alpha=0.45, linewidths=0,
               zorder=3, label="actual share")
    ax.set_ylim(bottom=0)
    _finish(fig, ax, path)


def raw_move(path: Path, title: str, weeks, ratio, implied: dict, pre: int, post) -> None:
    """Applications by week around one move, as a ratio to the pre-move average, against the
    level each assumed uplift implies for the post-move window."""
    fig, ax = _axes(title, "applications, ratio to the pre-move average")
    ax.axvspan(-pre - 0.5, -0.5, color=GRID, alpha=0.5, linewidth=0)
    ax.axvspan(post[0] - 0.5, post[1] + 0.5, color=GRID, alpha=0.5, linewidth=0)
    ax.axvline(0, color=MUTED, linewidth=0.9, alpha=0.6, label="move")
    ax.axhline(1, color=MUTED, linewidth=0.8)
    ax.plot(weeks, ratio, color=BLUE, linewidth=2, marker="o", markersize=4, label="actual")
    for (label, value), colour in zip(implied.items(), (ORANGE, AQUA)):
        ax.hlines(value, post[0] - 0.5, post[1] + 0.5, color=colour, linewidth=2,
                  label=f"implied by {label}")
        ax.annotate(f"{value:.2f}x", (post[1] + 0.6, value), color=MUTED, fontsize=8, va="center")
    ax.set_xlabel("weeks from the move (shaded: before and after windows)", color=MUTED, fontsize=9)
    _finish(fig, ax, path)
