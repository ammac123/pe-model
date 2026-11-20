# mortgage-elasticity

If we change our rate in a segment by some number of basis points, what happens to our
broker-channel applications and share over the next 12 weeks, and how sure are we?

One Bayesian model answers that. It gives one sensitivity per segment with an 80% range, a
12-week forecast for any proposed move, and a table that says how accurate those forecasts can be
on your data. Every number comes with a range; none should be read without it.

- Plain-language description of the model: [`docs/overview.md`](docs/overview.md)
- Derivations: [`docs/model_maths.pdf`](docs/model_maths.pdf) (source `docs/model_maths.tex`)

## Quick start

```bash
uv sync                                  # create .venv, install dependencies
uv run elasticity simulate               # synthetic panel, events, competitors -> data/sample/
IN="--panel data/sample/panel.csv --events data/sample/events.csv --competitors data/sample/competitors.csv"
uv run elasticity check-moves $IN --out outputs/sample   # no model: look at this first
uv run elasticity estimate    $IN --out outputs/sample   # about 2-3 minutes on a laptop
uv run elasticity forecast --fit outputs/sample/fit.pkl --segments "OO_PI_70-80" --headline -10
uv run elasticity export-excel --fit outputs/sample/fit.pkl --out outputs/sample/elasticity.xlsx
uv run elasticity charts --fit outputs/sample/fit.pkl --out outputs/sample/charts  # explanatory PNGs
uv run pytest                            # about 4 minutes; includes small model fits
```

For real data, put the three files in `data/raw/` (not tracked by git) and point the commands
there.

| command | what it does |
|---|---|
| `simulate` | writes synthetic inputs with known true sensitivities (`--u`, `--seed`, `--reactive`) |
| `check-moves` | raw-move check: applications before and after each past price move, against what 8%/bp and 3%/bp imply. No model. Run it first |
| `estimate` | fits the model and writes every output below |
| `forecast` | 12-week effect of a move: `--segments ALL` or `"A;B"`, `--headline`, `--discretionary`, `--competitor [SEGMENT:]LENDER=BPS` |
| `export-excel` | workbook with live formulas for scenario work |
| `charts` | explanatory charts as PNGs: `--segment`, `--move`, `--margin`, `--truth` (simulated data) |
| `validate events\|quiet\|placebo` | backtests; each refits the model several times |
| `sensitivity` | refits under other priors and settings |

`forecast`, `export-excel`, `charts`, `validate` and `sensitivity` read the `fit.pkl` that `estimate` wrote.

## Reading the outputs

### `segments.csv`: one row per segment

| column | how to read it |
|---|---|
| `u_p10`, `u_p50`, `u_p90` | **The sensitivity.** The percentage rise in our volume per 1bp cut, as a fraction (0.03 = 3% per bp). `u_p50` is the best single figure. There is a 4-in-5 chance the truth lies between `u_p10` and `u_p90`, and a 1-in-5 chance it does not |
| `prior_share` | **How much of that answer is judgment.** 0 means the data decided it. 1 means the data said nothing and the figure is just the starting assumption (3%/bp, range 1.2 to 7.4). Above about 0.5, treat the segment's number as mostly judgment |
| `share_now_*` | our fitted share of market lodgements in the latest week |
| `beta_*`, `T_bps` | the same sensitivity in the units the formulas use, and the benchmark's "softness" (see the overview) |
| `n_price_moves` | how many logged own price moves touched this segment |
| `thin_volume` | TRUE below 50 own lodgements a week. Week-to-week chance alone then moves volumes by more than ±5%, so read thin segments in groups |
| `conversion`, `conv_cv`, `conv_trend_per_period_pct` | applications per lodgement over the last 13 weeks, and whether that ratio is stable. If it trends, a change in lodgement share is not quite a change in applications |
| `sampler_warning`, `u_rhat`, `u_ess` | TRUE means the numerical fit for this segment is not trustworthy; see `run_log.txt` |

A segment with few price moves of its own still gets a usable figure, because segments that share
a purpose, repayment type or LVR band inform each other.

### `accuracy.csv`: what the tool can and cannot promise

For every segment and group (all, by purpose, by repayment type, by LVR band) and for moves of
±5, ±10 and ±25bp, it gives the half-width of the 80% range of the 12-week forecast: for
applications as a percentage (`apps_halfwidth_rel_*`) and for share in percentage points
(`share_halfwidth_pp_*`). `meets_5pct_*` and `meets_0p3pp_*` say whether that is inside the
business targets of ±5% and ±0.3 points.

There are two versions of every column:

- `*_price_effect`: our natural share stays where it is today. This is the "no competitor response
  and no market movement" condition. The range reflects only how well we know the sensitivity, the
  speed of response and today's level.
- `*_full_predictive`: natural share also keeps drifting as it has in the past, and weekly counts
  are noisy. This is the range to expect when comparing a forecast with what actually happens.

Expect the target to be met for small moves in large segments and in groups, and missed for 25bp
moves and thin segments. Nothing is tuned to make this table pass. An 80% range is exceeded one
time in five by design.

### Other files

- `global.csv`: the half-life of a price move in weeks (`h`), the typical uplift
  (`mu_u_pct_per_bp`), how investor (`a_inv`), interest-only (`a_io`) and LVR-band (`c_lvr[...]`)
  segments differ, the weekly drift of natural share (`sigma_g` bank-wide, `sigma_v` per
  segment) and the count-noise parameter `c`.
- `natural_share.csv` and `natural_share/*.png`: per segment and week, actual share, fitted share
  and **natural share**: the share we would have had if our price gap to the market had stayed
  where it started. Turnaround time, credit policy, service and brand all sit in it. Price moves
  and logged non-price changes are marked on the charts.
- `check_moves.csv` and `check_moves/`: one table and chart per past price move.
- `diagnostics_estimators.csv`: the model's `u` beside two older estimators (two-way fixed effects
  and event studies), each with its known bias. They are checks; they do not feed the estimate.
- `diagnostics_ppc.csv`: whether data simulated from the fitted model wobbles and persists from
  week to week as the real share does.
- `T_path.csv`: the benchmark temperature per iteration.
- `run_log.txt`: the configuration, data counts, exclusions, sampler diagnostics and every warning.
  **Read the warnings.** `UNRELIABLE RUN` and `SOFT-MIN TEMPERATURE DID NOT CONVERGE` mean the
  numbers should not be used as they stand.

### Charts and the explorer notebook

- [`notebooks/model_explorer.ipynb`](notebooks/model_explorer.ipynb): try price and competitor
  scenarios and see the model in pictures, with a plain-language reading of each chart and the
  formula behind it. Open it in VS Code or Jupyter with the `.venv` kernel and run from the top;
  cells headed `# SETTINGS` are the ones to edit. It reads `fit.pkl` and the csv tables of one run.
- `uv run elasticity charts` writes the same charts as PNGs. They come from
  `mortgage_elasticity/visuals.py`: how the model works (response curve, delay, benchmark), what
  it learned (sensitivity by segment, share history), scenarios in one, two and three dimensions
  (move size, segment by move, our move against a competitor's, move size against time), and how
  far to trust it (raw moves, backtests, precision, judgment checks). `--margin` adds a
  volume-against-margin illustration from a margin you supply; it is not a model output.

### `forecast`

Two ranges are printed, labelled:

- **Price-effect interval**: as in the accuracy table, natural share held.
- **Full predictive interval**: with drift and count noise. Its multiplier is the simulated
  scenario divided by the central no-change forecast.

Each gives the 12-week applications multiplier, 12-week share in percentage points and 12-week
applications, at the 10th, 50th and 90th percentile. The weekly path is in `forecast_weekly.csv`
(`--out`). A move phases in: at a two-week half-life only about 80% of it is felt on average over
the first 12 weeks. The steady-state multiplier is shown separately as the long-run effect and is
not the 12-week answer. The baseline continues any recent move that has not finished phasing in.

### `elasticity.xlsx`

- `segments`: `share_now`, `beta` at p10/p50/p90, `T_bps`, `h_p50`, conversion, `prior_share`.
- `benchmark`: each competitor's latest effective rate and weight. Type a new rate into the
  yellow `scenario_rate_pct` column to model a competitor move.
- `scenario`: type our headline and discretionary change (yellow cells, bp, + = dearer) per
  segment. The sheet works out the change in our gap to the market, the weekly multipliers, their
  12-week average at the low, middle and high sensitivity, and the long-run multiplier. Its range
  is the **80% range from sensitivity only**. It assumes past moves have fully phased in and uses
  the middle half-life, so it is narrower than `forecast`. A test checks it against `forecast`
  to within 0.5%.

## Inputs

Dates are ISO (`yyyy-mm-dd`) unless `date_format` is set in `config.py`. Schema errors, our
lodgements above the market's, negative or fractional counts, duplicate rows, unparseable dates
and unbalanced panels stop the run with a message. Weeks where we had no lodgements are kept.
Weeks where the market had none are left out and counted in the log.

**`panel.csv`**: one row per segment and week, every segment covering the same weeks.

| column | meaning |
|---|---|
| `segment` | `PURPOSE_REPAYMENT_LVRBAND`: purpose `OO` or `INV`, repayment `PI` or `IO`, LVR band `<=60`, `60-70`, `70-80`, `80-90` or `>90`. Example: `OO_PI_70-80`. The model pools sensitivity across these three dimensions, so the name must carry them |
| `period` | week END date |
| `lodg_mkt`, `lodg_us` | market and our lodgements, broker channel |
| `apps` | our applications |
| `headline_cum_bps` | cumulative change in our carded rate since the panel start (bp, + = dearer) |
| `discretionary_cum_bps` *(optional)* | cumulative change in our average discretionary discount (bp, + = less discount). If absent it is treated as zero, the log says so, and the model sees carded moves only |

Our rates must use the same convention as the competitor rates: the **day-weighted average over
the week**, not the rate on the week-end date. Mixing the two creates false gaps around every RBA
move.

**`events.csv`**: one row per change.

| column | meaning |
|---|---|
| `date` | when. A date inside a week counts in that week |
| `segments` | `ALL` for bank-wide, or segment names separated by `;` |
| `type` | `price` for an own price move, `other` for anything else: campaign, credit policy, commission, turnaround change |
| `description` | free text |

Log every `other` change you know of. Each one lets natural share jump once in the affected
segments from that week. A price move in the same week as a logged `other` change then teaches
the model little, which is the intent. An abrupt change you do not log is read as price response
if it coincides with a price move.

**`competitors.csv`** *(optional, but the model is much weaker without it)*: one row per lender,
segment and week, or a change log with `effective_date` instead of `period` (one row per rate
change, converted to the day-weighted weekly average).

| column | meaning |
|---|---|
| `period` (or `effective_date`), `segment`, `lender` | one row per lender; collapse to its best effective product first |
| `rate_pct` | carded rate in percent (e.g. 6.19) |
| `adj_bps` *(optional, default 0)* | value to the customer not in the rate, bp, + = cheaper. **This is where your manual adjustments go** (cashback, known discounting). They move the price variable directly, so keep a dated record of them |
| `eligible` *(optional, default TRUE)* | FALSE rows get no weight |

## Judgment settings (`config.py`)

Every judgment parameter is in `config.py` with a one-line description. The ones that matter most:

| setting | default | meaning |
|---|---|---|
| `prior_u_median`, `prior_u_log_sd` | 0.03, 0.70 | starting assumption for uplift per bp: middle 3%, 80% range 1.2 to 7.4%. 8%/bp sits at about the 92nd percentile: possible, not assumed |
| `kappa` | 0.5 | share of a discretionary-discount change that counts as price |
| `prior_h_median` | 2 weeks | starting assumption for the half-life of a move; the data update it |
| `softmin_T_bps` | `None` | `None` solves the benchmark temperature from the data; a number fixes it |
| `time_effect` | `"rw"` | smooth bank-wide trend. `"fe"` (free week effects) is for sensitivity runs only |
| `interval` | 0.80 | the range reported everywhere |

`uv run elasticity sensitivity` shows how far the answer moves when the prior middle is 2, 4 or
8%/bp, the temperature is halved or doubled, `kappa` is 0.3 or 0.7, week dummies replace the
smooth trend, or jumps are switched off.

## Validation

| command | what it shows |
|---|---|
| `check-moves` | whether applications moved after past moves as an assumed uplift implies |
| `validate events` | for each past move: refit without the 12 weeks after it, then compare the predicted 12-week lodgements with what happened, given the competitor rates and market volumes that occurred |
| `validate quiet` | 12-week forecasts from a new starting point every 13 weeks, on segments we did not reprice: the error floor from drift and sampling |
| `validate placebo` | a fake 10bp move at a quiet date should show no effect about 4 times in 5 |
| `sensitivity` | see above |
| `scripts/acceptance.py` | simulated data with known truth: does the 80% range contain the truth about 80% of the time? Results: [`docs/acceptance.md`](docs/acceptance.md) |

## Caveats

- The smooth bank-wide trend lets bank-wide price moves inform the estimate. It rests on one
  assumption: no other abrupt bank-wide shock lands in the same weeks as a bank-wide price move,
  unless you log it.
- Competitor discounts we cannot see blur the benchmark and bias sensitivity towards zero.
  `adj_bps` is the only defence.
- The benchmark temperature `T` and the sensitivity depend on each other and are jointly weakly
  identified. The `sensitivity` run reports how the answer moves when `T` is halved and doubled.
- Turnaround time is inside natural share by design. If a price cut causes a surge that slows
  turnaround, the estimated sensitivity is the net effect of both.
- Customers and brokers react when a change is announced, not only when it takes effect.
- The synthetic data is cleaner than real lodgement data.
- [`docs/rba_window_plan.md`](docs/rba_window_plan.md) is an older, unbuilt plan written against
  the previous five-estimate design.
