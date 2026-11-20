# RBA-window cross-check: implementation plan (not built)

Status: plan only, and written against the earlier five-estimate design. That design has been
replaced by one Bayesian state-space model (see `overview.md`), in which the smooth bank-wide trend
already lets the temporary gaps around RBA moves inform the estimate. `beta_fe`, `beta_pool`,
`beta_hier`, `shared_scale` and the `clean` flag below no longer exist; read this as background on
the identification idea, not as a plan against the current code.

## 0. Bottom line

- **Identification:** within each segment's own time series around each RBA decision (no period
  fixed effects), the *temporary* gap to r* that opens and closes while lenders pass the change on
  at different times, after netting out the window's level and a post-RBA step. Gap path only
  (needs `--competitors`).
- **Simulation** (in-memory, 40-60 seeds, 6 segments, 7 shared lenders, segment-invariant delays):
  - With simulate.py-sized gaps (0-3 week delays), `rba_scale`/truth median 0.97-1.00, sd 0.07-0.14.
    That is far better than beta_fe on the same variation (0.45-1.37).
  - Precision collapses when lenders' effective dates are close together: delays of 3-8 days plus
    realistic noise gave sd 0.67.
  - Very sensitive to the headline half-life hH: true 4 vs assumed 2 reads 0.61x; true 1 reads 1.35x.
  - Measures the response to *temporary, expected* gaps, which may be smaller than the response to
    a lasting gap.
- **Recommendation: conditional go.**
  - Run a pre-flight on the real files first (Phase 0). Build only if forecast se(rba_scale) <= 0.30
    with at least 3 usable cycles.
  - Disagreement says beta, hH or anticipation is off, not which.

## 1. Identification

Merge overlapping cycle windows into spans s, with W_s = [tau_first - P, tau_last + Q). For
segment k and t in W_s:

    y_kt = a_ks + sum_{c in s} gamma_kc 1[t >= tau_c] - beta_k x_kt + u_kt
    x_kt = L_hH(H_k)_t + kappa L_hD(D_k)_t - L_hH(r*_k - r*_k1)_t          (existing gap path)

M_s annihilates A_s = [1, step dummies]; x~ = M_s x_k, y~ = M_s y_k. Under full pass-through x
returns to its old level, so only the temporary bump in x~ identifies beta.

**Controlled for:**
- The segment level in each window.
- Lasting post-RBA shifts in our share (the steps). These cover refinance/purchase mix, broker
  routing and partial pass-through.
  - Sim: a +/-0.05 step tied to the RBA sign, with us always last, gave 1.17 without the step and
    0.99 with it.
  - The step costs little: sd 0.125 vs 0.128.
  - Trade-off: a lasting gap is absorbed by the step, by design.
- Market demand: share nets out market volume.

**Not controlled for:** all segments are treated at once, so there is no control segment.
Anything that hits our share in every segment in the window is uncontrolled: turnaround times
in refinance surges, broker attention, announcement effects.

**Threats:**
1. **Half-life mismatch.** For short gap pulses with geometric lags,
   E[rho^]/rho ~ phi(2 - phi^) / (phi^ + phi - phi^ phi), with phi = 1 - 2^(-1/h).
   - With hH assumed = 2: true hH 1 gives 1.32 (sim 1.35); true hH 4 gives 0.67 (sim 0.61).
   - So beta_rba tests beta and hH jointly.
2. **Temporary, expected gaps.** Variable-rate borrowers pay the rate at settlement, and brokers
   know competitors will follow. The response may be a fraction of the response to a lasting gap.
   This biases the estimate down and can't be tested in simulation.
   - So rba < pool is weak evidence; rba > pool is strong evidence.
3. **Our pass-through timing is a choice.**
   - Sim: timing tied to a common persistent shock showed no bias (1.00).
   - The mechanism tested was weak, so the threat isn't ruled out.
   - v2: give L(H) and L(r*) separate coefficients.
4. **Announcement effects.**
   - Sim: a random-sign 0.10 logit, 2-week shock raised sd from 0.12 to 0.16 and cut coverage to
     0.53. The placebo can't see it.
   - Diagnostics: estimate hikes and cuts separately (if we are consistently late, the gap sign
     flips between them, while an RBA-week confounder doesn't). Also report a sensitivity with x
     moved one period earlier.
5. **Other things in RBA weeks:** unseen competitor cashback is measurement error.
6. **Wrong T when competitors run specials.**
   - Sim with T = 1/beta0: median 0.92, coverage 0.70.
   - With the true T: median 1.02, coverage 0.97.
7. **Fixed-rate segments** don't track the RBA: exclude them through config, and flag a segment
   whose own move doesn't match the RBA change.

**Pooling:**
- One shared ratio rho = beta_k / beta0_k, with beta0 = u/(1 - s0), the same base as
  `shared_scale`, so the two are directly comparable.
- Per-segment beta_rba is a diagnostic only.
- The placebo uses the same calendar offset for every segment, because pooling across segments
  doesn't average out common shocks.

**Monthly hiking runs:** 10 cycles 4-5 weeks apart merge into one span of about 54 weeks, and the
estimator still works.
- One step per cycle: median 0.97, sd 0.052, coverage 0.82 at z = 1.645 and 0.88 at z = 2.
- Default: one step per cycle; one step per span as a sensitivity.

## 2. Inputs

New optional `--rba rba.csv`:
- Required: `decision_date`, `change_bps` (signed, + = hike).
- Optional: `note`.
- Supplied rather than derived from the competitor log: derivation is fragile with competitor
  specials, and the dates are public, few and auditable.

Snapping: tau_c = the first period end on or after `decision_date`, as in events.py.

Validation:
- Missing columns, unparseable dates or duplicate dates are errors.
- `change_bps == 0` rows are ignored, with a message.
- All |change_bps| < 1: raise "use bps, not percent". |change_bps| > 100: warn.
- Cycles off the panel edge are dropped and counted as `window_edge`.
- For each cell, compare our headline change over the span with sum(change_bps). If the difference
  exceeds `rba_passthrough_tol_bps`, flag "partial/no pass-through". This catches fixed-rate
  segments.

## 3. Estimator

1. Use the final run's df and dfx, with r* at the re-solved T, and the grid's best kappa/hH/hD.
2. Windows: P = `rba_pre` = 6, Q = `rba_post` = 8 (Q >= the slowest lender's lag + 2 hH). Merge
   overlapping windows into spans.
3. Exclude a cell (span x segment) if:
   - its segment is in `rba_exclude_segments`;
   - there is an events.csv row for that segment inside the window but outside
     [tau_c - 1, tau_c + 4) (logged pass-through rows don't count);
   - max|x~| < `min_dx_bps` ("small_gap").
4. Pooled ratio: rho^ = -sum_{s,k} beta0_k <x~, y~> / sum_{s,k} beta0_k^2 <x~, x~>.
   Per segment: beta^_k = -sum_s <x~, y~> / sum_s <x~, x~>.
5. SE by residual placebo:
   - Residualize the panel: y. = y + rho^ beta0 x.
   - For b = 1..500 (seed 0): draw one offset per span, at least 3 periods from the real date, the
     same for every segment. Recompute rho_b on the shifted residuals with the real x~.
   - se = sqrt(mean rho_b^2), the RMS about 0.
   - If there are fewer than 10 offsets, fall back to Newey-West (L = 4) on the pooled score, and
     warn.

   Why this method, from simulation:

   | SE method | 90% coverage |
   |---|---|
   | Quiet-window placebo (events.py style) | 0.57-0.82 |
   | Newey-West | 0.55-0.68 |
   | Residual placebo, z = 1.645 | 0.80-0.90 |
   | Residual placebo, z = 2 | 0.88-0.97 |
6. Sensitivity rows in `rba_summary`:
   - all; hikes only; cuts only;
   - each hH in `hH_grid`;
   - one step per span; no step;
   - x moved one period earlier.

   Don't flag the spread across hH values: about 0.68 / 1.0 / 1.33 is expected even when hH is right.
7. Comparison:
   - z = (rho^ - shared_scale) / sqrt(se^2 + shared_scale_sd^2). Warn if |z| > `rba_flag_z` = 2.
   - Also print the median of beta_fe / beta_prior, and of beta_hier / beta_prior with `--bayes`.
   - Warn if hikes and cuts differ by more than 2 sqrt(se_h^2 + se_c^2).

**Sheets:**
- `rba_windows`, one row per cycle x segment: `cycle_id, decision_date, change_bps, period,
  span_id, segment, used, excluded_reason, our_move_bps, our_half_week, rstar_move_bps,
  rstar_half_week, lead_lag_weeks, raw_peak_gap_bps, peak_gap_bps, sum_x2, beta_cell,
  beta_cell_se, n_periods, passthrough_flag`.
- `rba_summary`: `variant, rba_scale, se, n_cycles, n_cells, n_placebo_offsets, se_method`.

**`parameters`, appended:** `rba_scale, rba_scale_se, beta_rba, beta_rba_se, n_rba_cycles,
rba_check` (`ok` / `disagree` / `n/a`). These are always present: NaN or n/a when the check isn't run.

Example output:

    RBA cross-check (NOT used in Excel): rba_scale 0.92 ± 0.18 (3 cycles, 17 windows, 74 placebo
    offsets) vs shared_scale 1.04 ± 0.10 (z=-0.6, consistent); median beta_fe/prior 0.71;
    hikes 0.88 / cuts 0.97; hH grid 1/2/4 -> 1.21/0.92/0.63

## 4. Files

- **New `src/mortgage_elasticity/rba.py`** (about 220 lines): `load_rba`, `build_spans`, `cells`,
  `pooled_ratio`, `per_segment`, `residual_placebo_se` and `run(...)`, which returns the parameter
  columns, the windows and summary sheets, and messages.
- **`config.py`:** `rba_pre=6`, `rba_post=8`, `rba_step="per_cycle"`, `rba_exclude_segments=()`,
  `rba_passthrough_tol_bps=5.0`, `rba_placebo_draws=500`, `rba_flag_z=2.0`, `rba_min_cycles=3`.
- **`cli.py`:** a `--rba` flag; one block after the final `_fit`; 2 sheets and a meta row; skipped
  with a message when `--competitors` is absent.
- **`simulate.py`:** `_market` returns its steps; with `competitors=True`, also write `rba.csv`.
- **New `tests/test_rba.py`:**
  - Validation errors; span merging.
  - Noiseless exact recovery, with and without a step.
  - Invariance to segment constants and to an RBA-sign step.
  - Pure noise gives |rho^| < 3 se.
  - Full-pipeline recovery on 5 seeds, within 35%.
  - Without competitors, NaN columns.
- **Docs:**
  - README: input table, sheets and columns, and "never use beta_rba in Excel".
  - overview.md step 4: one paragraph.
  - model_maths.tex: a new subsection in section 4 (FWL with step dummies, the half-life lemma,
    temporary vs lasting response, residual-placebo SE, limits).

## 5. Validation gates

Results so far (median rho^/rho, sd):
- Week delays: 0.99, 0.07 (peak perceived gap 6.6 bp).
- Day delays of 3-14 days: 0.97-1.00, 0.12-0.14 (peak perceived gap 3.3 bp).
- Day delays with noise 0.15 and a common AR(1) shock: sd 0.28-0.30.
- Delays of 3-8 days: sd 0.67.
- 6 cycles over 156 weeks: sd 0.08. 1 cycle: sd 0.23.
- Relative SE scales with 1/beta: the check is weakest when the true beta is below the prior.

Gates:
- **G0, real data (the main gate):** placebo SE <= 0.30 at rho = 1, with at least 3 usable cycles.
- **G1:** median se <= 0.30 in the realistic sim, with noise matched to the real residuals.
- **G2:** |median - 1| <= 0.15 at the correct hH.
- **G3:** coverage at z = 2 within 0.85-0.98, and at 1.645 at least 0.80.
- **G4:** flag false alarms <= 10% when the prior is right; fires at least 50% of the time when u is
  off by 50%.
- **G5:** report the hH +/-1 step, announcement and step stresses, with no threshold.

Decision:
- G0 or G1 fails: don't ship.
- G1 passes but G3 or G4 fails: ship as a sheet-only diagnostic, with no `parameters` columns.

## 6. Size

- **Phase 0 (half a day):** a scratch pre-flight script on the real files, giving the forecast SE
  and gap sizes per cycle. Do this first.
- **MVP:** about 1.5 days plus half a day of validation.
- **Deferred:** separate coefficients on L(H) and L(r*); competitor rates dated by announcement;
  automatic detection of missing RBA rows; choosing hH from the windows; per-segment inference.
