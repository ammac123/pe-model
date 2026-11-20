"""Every judgment parameter and pipeline setting. Edit these, not the code."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # --- sensitivity prior (uplift per bp: fractional rise in our volume per 1bp cut) ---
    prior_u_median: float = 0.03  # middle of the prior: 1bp cheaper brings about 3% more volume
    prior_u_log_sd: float = 0.70  # how unsure we are: 0.70 puts 80% of the prior at 1.2-7.4%/bp
    prior_dim_sd: float = 0.30  # how far investor / interest-only sensitivity may sit from the rest
    prior_lvr_sd: float = 0.30  # how far LVR bands may differ from each other in sensitivity
    segment_residual: bool = False  # let each segment differ beyond purpose / repayment / LVR
    segment_residual_sd: float = 0.15  # size of that extra per-segment difference, if switched on

    # --- timing and discount shading ---
    kappa: float = 0.50  # share of a discretionary-discount change that customers see as price
    prior_h_median: float = 2.0  # weeks for half of a price change to be felt (prior middle)
    prior_h_log_sd: float = 0.50  # how unsure we are about that half-life
    h_bounds: tuple[float, float] = (0.5, 12.0)  # half-life is kept between these (weeks)

    # --- natural share ---
    prior_alpha_sd: float = 1.0  # how far a segment's base level may sit from its average share
    prior_sigma_g: float = 0.05  # typical weekly drift of the bank-wide trend (logit units)
    prior_sigma_v: float = 0.05  # typical weekly drift of each segment's own trend (logit units)
    time_effect: str = "rw"  # "rw" = smooth bank-wide trend; "fe" = free week effects (sensitivity
    #                          runs only: free week effects absorb every bank-wide price move)
    prior_fe_sd: float = 0.50  # spread of the free week effects when time_effect = "fe"
    jumps: bool = True  # allow a one-off level shift at each logged non-price change
    prior_jump_sd: float = 0.30  # typical size of such a shift (logit units; 0.3 is about 30%)
    covariates: tuple[str, ...] = ()  # extra panel columns to control for; none by design
    prior_gamma_sd: float = 0.20  # typical effect of one standard deviation of a covariate

    # --- count noise ---
    prior_c_median: float = 1000.0  # beta-binomial concentration: large = plain binomial noise
    prior_c_log_sd: float = 1.5  # how unsure we are about it

    # --- soft-min benchmark ---
    softmin_T_bps: float | None = None  # None = solve T = 1/beta per segment; a number fixes it
    T_bounds: tuple[float, float] = (2.0, 60.0)  # T is kept between these (bp)
    T_tol: float = 0.05  # stop the T loop when no segment's T moves by more than this fraction
    T_max_iter: int = 6  # give up (and warn) after this many T iterations
    T_tune: int = 500  # warm-up steps of each short fit inside the T loop
    T_draws: int = 500  # draws per chain of each short fit inside the T loop
    T_chains: int = 2  # chains of each short fit inside the T loop
    exclude_lenders: tuple[str, ...] = ()  # competitors left out of the benchmark
    comp_ffill_periods: int = 8  # carry a competitor's last rate forward at most this many weeks
    min_lenders: int = 3  # warn if a segment has fewer competitors than this

    # --- sampler ---
    chains: int = 4  # independent sampling runs of the full fit
    tune: int = 1000  # warm-up steps per chain
    draws: int = 2000  # draws kept per chain (1000 leaves R-hat hovering at the 1.01 limit)
    target_accept: float = 0.95  # sampler caution: higher is slower and safer
    seed: int = 20261120  # fixed so a rerun on the same data gives the same answer
    nuts_sampler: str = "auto"  # "auto" = nutpie if installed, else numpyro, else PyMC's own
    max_rhat: float = 1.01  # a run is flagged unreliable above this
    min_ess: float = 400.0  # ... or if a key parameter has fewer effective draws than this
    prior_draws: int = 2000  # prior draws for the prior check and prior_share

    # --- reporting ---
    interval: float = 0.80  # width of every reported range (0.80 = 10th to 90th percentile)
    horizon: int = 12  # forecast length in weeks
    baseline_window: int = 13  # trailing weeks for market volume and the conversion ratio
    thin_lodgements: float = 50.0  # flag a segment as thin below this many own lodgements a week
    accuracy_moves_bps: tuple[float, ...] = (-25, -10, -5, 5, 10, 25)  # moves in the accuracy table
    target_apps_rel: float = 0.05  # accuracy target: applications within +/-5%
    target_share_pp: float = 0.3  # accuracy target: share within +/-0.3 percentage points
    lvr_bands: tuple[str, ...] = ("<=60", "60-70", "70-80", "80-90", ">90")  # LVR band labels
    date_format: str | None = None  # e.g. "%d/%m/%Y"; None = ISO (yyyy-mm-dd)

    # --- raw-move check and diagnostics ---
    move_pre_weeks: int = 4  # weeks before a move used as its "before" average
    move_post_weeks: tuple[int, int] = (3, 10)  # weeks after a move used as its "after" average
    check_u: tuple[float, ...] = (0.08, 0.03)  # uplifts whose implied response the check shows
    event_pre: int = 4  # event-study diagnostic: weeks before
    event_post: int = 8  # event-study diagnostic: weeks after
    min_dx_bps: float = 1.0  # event-study diagnostic: skip moves smaller than this
    ppc_draws: int = 400  # simulated histories for the posterior predictive checks

    # --- validation and sensitivity runs ---
    val_chains: int = 4  # sampler settings of each validation / sensitivity refit; their
    val_tune: int = 1000  # diagnostics are logged but, being shorter runs, not held to the
    val_draws: int = 1000  # max_rhat / min_ess limits above
    quiet_first_origin: int = 52  # first forecast origin of the quiet-period backtest (week no.)
    quiet_step: int = 13  # weeks between quiet-period forecast origins
    placebo_runs: int = 10  # number of fake moves tried
    placebo_bps: float = 10.0  # size of each fake move
    placebo_coef_sd: float = 0.10  # prior spread of the fake move's coefficient (per bp)
    sens_prior_medians: tuple[float, ...] = (0.02, 0.04, 0.08)  # prior middles to refit with
    sens_T_factors: tuple[float, ...] = (0.5, 2.0)  # T is multiplied by these
    sens_kappas: tuple[float, ...] = (0.3, 0.7)  # discount shadings to refit with

    def __post_init__(self):
        if not (self.prior_u_median > 0 and self.prior_u_log_sd > 0):
            raise ValueError("config: need prior_u_median > 0 and prior_u_log_sd > 0")
        if self.time_effect not in ("rw", "fe"):
            raise ValueError('config: time_effect must be "rw" or "fe"')
        if not 0 < self.interval < 1:
            raise ValueError("config: interval must be between 0 and 1")

    @property
    def q(self) -> tuple[float, float, float]:
        """Lower, middle and upper quantile of the reporting interval."""
        return ((1 - self.interval) / 2, 0.5, (1 + self.interval) / 2)
