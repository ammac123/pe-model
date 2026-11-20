# Acceptance runs on simulated data

Run on 20 November 2026 with `uv run python scripts/acceptance.py` (60 full estimates, each
including the benchmark-temperature loop). Raw results: `outputs/acceptance/runs.csv` and
`summary.csv` (not tracked by git; rerun to regenerate).

Each dataset has 20 segments (8 or so thin) by 104 weeks, 12 own price moves (6 bank-wide
including two RBA pass-throughs, 6 in groups of segments), competitor specials, and three logged
non-price changes, one of them in the same week as a price move. The true sensitivities are known.

## Results

| case | datasets | 80% range contains the true `u_k` | error in typical uplift | bias of `u_p50` | temperature converged | flagged unreliable |
|---|---|---|---|---|---|---|
| true uplift 3%/bp | 10 | **81%** of 200 | 3.8% | −2.5% | 10 of 10 | 2 |
| true uplift 8%/bp | 10 | **83%** | 2.2% | −1.3% | 10 of 10 | 1 |
| true uplift 1.5%/bp | 10 | **83%** | 5.2% | −3.7% | 10 of 10 | 3 |
| 3%/bp, reactive repricing on | 10 | **86%** | 4.8% | −4.2% | 10 of 10 | 0 |
| 3%/bp, segments off the pooling pattern | 10 | **79%** | 3.0% | −2.7% | 10 of 10 | 2 |
| same, with `segment_residual` on | 10 | **84%** | 3.2% | −2.9% | 10 of 10 | 1 |

- *Contains the true `u_k`*: share of segment-runs whose 10th-to-90th percentile range holds the
  truth. The target was 70 to 90%.
- *Error in typical uplift*: |posterior median of exp(`mu_u`) ÷ truth − 1|, averaged over
  datasets. The target was under 25%.
- *Bias*: the median over segments of `u_p50` ÷ truth − 1, averaged over datasets. With reactive
  repricing the target was within ±20%.

## Against the criteria in the brief

| criterion | result |
|---|---|
| 3%/bp: coverage 70–90%, typical uplift within 25% | **Pass** (81%, 3.8%) |
| 8%/bp and 1.5%/bp, reported separately | **Pass** (83%, 2.2%; 83%, 5.2%). The prior, centred on 3%, does not dictate the answer when the data are informative |
| Reactive repricing: bias within 20% | **Pass** (−4.2%; worst single dataset −11%) |
| Unit tests | **Pass** (53 tests) |
| Full estimate on 20 segments × 104 weeks under 30 minutes | **Pass** run on its own: 2 to 6 minutes on a 14-core laptop. Inside the acceptance batch, with three estimates and other jobs sharing the machine, the slowest took 34 minutes |

## What to take from it

- **The ranges are honest on data that follow the model.** About 4 in 5 true values fall inside
  the 80% range in every case.
- **There is a small downward lean**, 1 to 4%, in every case. It is well inside the reported
  ranges but it is consistent in sign.
- **Coverage varies between datasets**, from 50% to 100% of segments in a single dataset. Errors
  are shared across segments (they move together through the pooled parameters), so one dataset is
  either mostly inside or noticeably out. Your data are one dataset.
- **Reactive repricing did not bias the estimate.** Cutting price in segments whose share had
  slipped (up to six extra cuts) left the bias at −4%.
- **Segments off the pooling pattern.** When true sensitivities departed from the additive
  purpose / repayment / LVR pattern by up to 10%, coverage fell only to 79%, and switching
  `segment_residual` on restored it to 84%. Larger departures would cost more: the default model
  has no way to represent them.
- **The "unreliable run" flag fires on fits that look healthy about 15% of the time.** In all nine
  flagged runs the cause was a handful of divergent sampler steps (1 to 7 in 8,000 draws), with
  R-hat and effective sample size fine. Seven of the nine were as accurate as the rest; the two
  with 5 and 7 divergences were among the weaker ones (50% and 60% coverage). Treat a flag as
  "rerun with a higher `target_accept` or another seed, and compare", not as proof of failure.
- **At 1.5%/bp the temperature hits its 60bp ceiling** (the consistent value is 61 to 66bp). The
  effect on the estimates was small.

## What this does not show

The simulated data follow the model's own assumptions: a smooth bank-wide trend, every non-price
change logged, competitor discounts fully visible, one half-life. It tests the estimator and the
code, not those assumptions. On real data the backtests (`validate events`, `validate quiet`,
`validate placebo`) and the `sensitivity` run are the relevant checks.

One design choice in the simulator matters for reading the 8%/bp case. Simulated repricing leans
back towards the market, and move sizes shrink when the true uplift is above 3%/bp. Without that,
at 8%/bp a run of 10 to 15bp moves in one direction takes a segment to most of the market, which
no lender's data look like, and the pooling pattern breaks down.
