# mortgage-elasticity

Estimation pipeline for the binary logit share model. Python estimates the parameters;
Excel reads `parameters.xlsx` and applies

```
beta = u / (1 - s0)                          # or an estimated beta: see "Which beta to use"
multiplier = 1 / (s0 + (1 - s0) * EXP(beta * x))
apps = A0 * multiplier
```

## Quick start

```bash
uv sync                                   # create .venv, install deps
uv run elasticity simulate --competitors  # synthetic panel, events, competitors -> data/sample/
uv run elasticity estimate --panel data/sample/panel.csv --events data/sample/events.csv \
    --competitors data/sample/competitors.csv --out outputs/sample
uv run pytest                             # model, inputs, estimators, pooling, end-to-end tests
```

Real data (`--competitors` is optional; without it x is our own price move, see below):

```bash
uv run elasticity estimate --panel data/raw/panel.csv --events data/raw/events.csv \
    --competitors data/raw/competitors.csv --out outputs
```

Options: `--no-grid` (skip the kappa/half-life search and use the priors in `config.py`),
`--bayes` (also fit the PyMC hierarchical model, `beta_hier`, the main estimate with competitor
data; first run `uv sync --extra bayes`, and use `uv run --extra bayes elasticity ...`).
`simulate --reactive` also cuts price whenever volume falls short. It logs those cuts as
`clean=FALSE`, as if you had vetoed them. If you log them as usable, `beta_post` / `beta_pool` come
out about 17% low.

Exploratory work: `uv add --dev jupyter` then `uv run jupyter lab`.

## Inputs

Dates are ISO (`yyyy-mm-dd`) unless `Settings.date_format` is set (e.g. `"%d/%m/%Y"`).
Unparseable dates, blanks, duplicates and unbalanced panels are errors, not warnings.

**`panel.csv`**: one row per segment x period (weekly or monthly, no gaps, every segment the same
periods).

| column | meaning |
|---|---|
| `segment` | segment id, identical definition in our data and market data (no `[` or `]`) |
| `period` | period END date |
| `lodg_mkt` | market lodgements in the segment |
| `lodg_us` | our lodgements in the segment |
| `apps` | our applications (used for the conversion check) |
| `headline_cum_bps` | cumulative headline rate change since panel start (bps, + = dearer) |
| `discretionary_cum_bps` | cumulative change in average discretionary discount given (bps, + = dearer, i.e. less discount) |

Our rates must use the same convention as the competitor rates: the **day-weighted average over
the period**, not the rate on the period end date. Mixing the two creates false gaps around every
RBA move.

**`events.csv`**: one row per repricing.

| column | meaning |
|---|---|
| `segment`, `period` | where and when. An event dated inside a period counts in that period (the first period end on or after the date); events after the last period are dropped |
| `reason` | stated reason (funding cost, competitor, volume shortfall, ...) |
| `other_changes` *(optional)* | anything else that changed in the segment around the move: campaign, broker incentive, credit policy, turnaround... Blank or `none` if nothing. A move with other changes is reported but doesn't update beta: its share response can't be put down to price |
| `clean` *(optional, default TRUE)* | set FALSE to keep a move out of the beta update (TRUE/FALSE, 1/0, yes/no). You don't need to classify motives |
| `d_headline_bps`, `d_discretionary_bps` *(optional)* | size of the move (informational only) |

Moves made in every segment at once (e.g. passing on an RBA change) can be left out: they have no
control segments, and they already enter through x.

A campaign or other change with **no** price move can be logged too (give a `reason`, put it in
`other_changes`, sizes 0). It produces no estimate, but keeps those weeks out of the noise
estimate, and keeps that segment out of the comparison group for other moves at the time.

**`competitors.csv`** *(optional, `--competitors`)*: each eligible competitor's rate, one row per
period x segment x lender. Alternatively, a change log with `effective_date` instead of `period`
(one row per rate change). It is converted to the day-weighted average rate over each period. A
lender's rate before its first change row is unknown.

| column | meaning |
|---|---|
| `period` (or `effective_date`), `segment`, `lender` | one row per lender; collapse to its best effective product first (duplicates are an error) |
| `rate_pct` | the lender's rate in percent (e.g. 6.19) |
| `adj_bps` *(optional, default 0)* | value to the customer not in the rate (cashback etc.), bps; + lowers the effective rate |
| `eligible` *(optional, default TRUE)* | FALSE rows are ignored |

Settings: `exclude_lenders`, `comp_ffill_periods` (how long a missing rate is carried forward),
`min_lenders` (warn below), `softmin_T_bps`.

## The price the customer sees (x)

- **Gap model** (with `--competitors`): the market benchmark is the soft-min of the competitors'
  effective rates, `r* = -T log sum_j exp(-(rate_j - adj_j) / T)` (the logit inclusive value).
  `T` = `1 / beta` per segment: the run starts at `1 / beta_prior`, then re-solves once at
  `1 / beta_fe` and estimates again (`Settings.resolve_T`; `Settings.softmin_T_bps` fixes one
  `T` for all segments, `0` = cheapest lender). `beta_fe` is used because it doesn't depend on the
  event log, which is thin and mostly mixed-motive; a segment with no positive `beta_fe` falls back
  to `1 / beta_pool`, then `1 / beta_prior` (with a warning).
  A lender entering lowers r*, so a changing number of competitors moves r* even when no rate
  moves (the pipeline warns). Then `x = lag(headline) + kappa lag(discretionary) - lag(r* - r*_first)`.
- **Move model** (no competitor file): x is our own move, minus its mean across segments in each
  period (`Settings.demean_x_by_period`). This stops RBA moves that everyone passes on from
  looking like price changes that didn't hurt volume. It also removes our own bank-wide moves,
  which can't be told apart from market moves without competitor data. Competitors' own moves are
  invisible here, so expect noisier betas than the gap model.

## Pipeline (`elasticity estimate`)

| step | module | what it does |
|---|---|---|
| 1-2 | `panel.py` | validate inputs, share, logit share, baseline s0, conversion check, competitor benchmark r*, x |
| 3 | `config.py` | judgment priors: u = 8%/bp, prior sd, segment sd, kappa, half-lives |
| 4a | `fe.py` | two-way fixed effects `y = alpha_k + delta_t - beta_k x`, grid over kappa / half-lives |
| 4b | `events.py` | difference-in-differences per event vs segments not repriced in the window |
| 5a | `pool.py` | prior -> posterior per segment (`beta_post`) and pooled across segments (`beta_pool`), using only usable events (no other changes, not vetoed) |
| 5b | `bayes.py` | optional hierarchical panel model in PyMC (`beta_hier`, the main estimate with competitor data) |
| 6 | `alpha.py` | natural position over time (inversion + local-level Kalman), beta held fixed at `beta_post` |
| 7 | `cli.py` | writes `parameters.xlsx` / `.csv` |

## Outputs (`<out>/parameters.xlsx`)

- `parameters`: per segment `s0`, `beta_prior`, `n_clean_events` (usable events), `beta_post`
  (+ 90% interval), `u_post`, `beta_fe`, `prior_weight` (how much of the answer is still
  judgment), `kappa`, half-lives (`beta_hier` + `beta_hier_sd` too if `--bayes` was used). Then, appended: `beta_pool` (+ sd, 90% interval),
  `u_pool`, `shared_scale` (+ sd), `own_weight`, `x_basis` (`move`/`gap`), `softmin_T_bps`,
  `r_star_now_pct`, `n_lenders_now`, `x_drift_s0_window`. **Excel contract:** existing columns are
  never renamed, removed or reordered; new ones only go on the end.
- `event_estimates`: every estimable event with its beta estimate and SE, snapped period,
  `own_events_in_window`, `other_changes`, `flagged_clean` (your flag as given) and `clean`: the
  event is **usable** for updating beta (`n_clean_events` counts these). The run prints how many
  events were dropped and why.
- `timing_grid`: SSR across kappa / half-life combos. The grid winner is used only if it cuts SSR
  by at least `grid_min_gain` (10%) against the priors; otherwise the priors are kept.
- `conversion_check`: if `conv_trend_per_period_pct` or `conv_cv` is large, lodgement share is a poor
  proxy for application share in that segment.
- `natural_share`: natural share by segment over time (Kalman-smoothed with 24+ periods, filtered below that).
- `benchmark` (gap model): last period's competitor rates and each lender's weight in r*.
- `meta`: input paths, period range, x basis, T, flags, run time.

## Which beta to use

Most own repricings are mixed-motive (strategy, volume, competitors), and only a handful are
usable events. So the panel estimates, which learn from every change in our price *relative to
competitors*, should carry the weight. The event-based ones are the u = 8% judgment updated by
logged moves. With few moves logged, they mostly restate the judgment. If you log every mixed-motive
move (`clean` defaults to TRUE), they also pick up the bias from moves made because share had
slipped: about -16 to -18% in simulation.

With competitor rates (`--competitors`):

- `beta_hier` (`--bayes`): **the main estimate.** A panel model that lets each segment's level
  drift, so a move made because share had already fallen is measured against where share had
  fallen to, not against its long-run average. In simulation it stayed close to the truth when we
  cut price whenever share fell: 0.95-1.12 × truth across 4 runs with competitor data, while
  `beta_fe` was 0.77-1.11. Its 90% intervals covered the truth 22 of 24 times without those cuts
  and 19 of 24 with them.
  - The run warns if sampling looks unreliable (r_hat > 1.01, ess < 400, or any divergences): check
    the `hierarchical` sheet.
  - With frequent reactive cuts the warning fired in every simulation, yet the estimates were good
    (r_hat at most 1.02). So treat a mild warning as "check it against `beta_fe`". If r_hat is above
    about 1.05, don't use it.
- `beta_fe`: **the cross-check.** It learns from the same variation without the drifting level.
  - Competitors' moves alone are enough for it: in simulation with none of our own moves it came
    out at 1.02 × truth.
  - It is biased towards zero when our moves react to share and the shortfalls persist: about
    0.5 × truth in 3 of 4 such simulations.
  - So if it is well below `beta_hier`, suspect that.
  - `beta_fe_se` (HC1) is badly over-confident when share drifts. Don't build intervals from it.
- `beta_pool` / `beta_post`: the judgment, updated by the few usable events.
  - Use `shared_scale` as a test of the u = 8% assumption: if its 90% interval
    (±1.645 × `shared_scale_sd`) excludes 1, the assumption is off across the board.
  - `beta_pool` pools usable events from all segments; with none it is exactly the prior.
    `beta_post` uses each segment's own events only.
  - Simulation with 0-1 usable events per segment (20 seeds): when u was ~50% too high, relative
    RMSE was prior 75%, `beta_post` 61%, `beta_pool` 38%. When u was right on average but segments
    differed by ±30-50%, it was `beta_post` 28%, `beta_pool` 30-33%.
  - `Priors.segment_sd_frac` (default 0.15) decides how far a segment may differ from the shared
    pattern. Raise it (e.g. 0.25) if segments plausibly differ a lot, e.g. fixed vs variable.

Without competitor rates, x is only our own moves, so every estimate rests on mixed-motive
decisions. Prefer `beta_hier` for the reason above, and treat all of them as rough.

If the estimates disagree a lot, look at `event_estimates` and the `hierarchical` sheet before
trusting any of them. The run warns if any `beta_post`, `beta_pool`, `beta_fe` or `beta_hier` is <= 0, or if x moved
more than 5bp inside the s0 window (`x_drift_s0_window`): Excel's multiplier assumes s0 is the share at today's price.

## Caveats

- Identification:
  - Moves common to every segment are absorbed by the period effects. This includes pass-through
    timing that is the same in all our segments, e.g. "we passed the RBA cut on 10 days after the
    majors". Only timing differences that vary by segment (ours or competitors') inform beta_fe.
    An RBA-window cross-check is planned but not built: see
    [`docs/rba_window_plan.md`](docs/rba_window_plan.md).
  - Competitor discounts we can't see are measurement error in r*, which biases beta towards zero.
  - Customers and brokers react when a change is announced, not only when it takes effect. Rates
    are entered as at their effective date, so the weeks between announcement and effective date
    can show odd share moves that aren't price responses.
- Which events update beta (`clean` in `event_estimates`): an event is biased if something the model
  can't see moved share in the same weeks as the price. Every move is partly strategy, volume and
  competitor reaction, so we don't ask for a motive, and there is no data screen. A check on share
  drifting before each move was tried and dropped: in simulation it barely told moves made after
  share slipped (4-5% flagged) from random ones (3%). Log simultaneous changes (campaigns, broker
  incentives) in `other_changes`: any entry makes the event unusable. `clean=FALSE` is a veto.
- The event SE comes from placebo windows (same DiD on windows with no event and no price move),
  so it captures drift and autocorrelation. It falls back to first differences if there are fewer
  than 10 such windows. It adds a heterogeneity term when the controls' x also moved.
- HC1 standard errors are used instead of clustering by segment, because there are too few segments
  for clustered SEs to be reliable.
- The synthetic data is cleaner than real lodgement data (2 years weekly, known timing).
- Don't re-estimate beta inside the alpha filter (step 6).

## Documentation

- [`docs/overview.md`](docs/overview.md): plain-language, step-by-step description of the model.
- [`docs/model_maths.pdf`](docs/model_maths.pdf) (source `docs/model_maths.tex`): technical
  derivations. Section N expands step N of the overview.
