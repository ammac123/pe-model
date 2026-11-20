"""Judgment priors and pipeline settings. Edit these, not the code."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Priors:
    u: float = 0.08  # current judgment uplift: +8% apps per bp at baseline
    prior_sd_frac: float = 0.30  # prior sd on beta as a fraction of the prior mean (governance call)
    segment_sd_frac: float = 0.15  # pooling: how far a segment's beta/prior ratio may sit from the
    #                                shared ratio (must be < prior_sd_frac)
    kappa: float = 0.50  # discretionary shading
    h_headline: float = 2.0  # half-life of headline moves (periods)
    h_discretionary: float = 8.0  # half-life of discretionary moves (periods)


@dataclass(frozen=True)
class Settings:
    s0_window: int = 13  # trailing periods used for baseline share
    event_pre: int = 4  # periods before an event
    event_post: int = 8  # periods after an event (long enough to cover the lag)
    min_dx_bps: float = 1.0  # ignore events whose perceived price moved less than this
    kappa_grid: tuple[float, ...] = (0.3, 0.5, 0.7)
    hH_grid: tuple[float, ...] = (1.0, 2.0, 4.0)
    hD_grid: tuple[float, ...] = (4.0, 8.0, 16.0)
    grid_min_gain: float = 0.10  # use the grid winner only if it cuts SSR this much vs priors
    min_periods_alpha_mle: int = 24  # below this, alpha filter uses the judgment q/r below
    q_over_r: float = 0.1
    date_format: str | None = None  # e.g. "%d/%m/%Y"; None = ISO (yyyy-mm-dd)
    demean_x_by_period: bool = True  # move path only: strip moves common to all segments from x
    softmin_T_bps: float | None = None  # gap path: None = 1/beta per segment, 0 = hard min
    resolve_T: bool = True  # with T = 1/beta: start at 1/beta_prior, re-solve once at 1/beta_fe
    #                         (not beta_pool: the event log is thin); falls back per segment
    exclude_lenders: tuple[str, ...] = ()
    comp_ffill_periods: int = 8  # carry a competitor's last rate forward at most this many periods
    min_lenders: int = 3  # warn if fewer competitors than this in a segment
