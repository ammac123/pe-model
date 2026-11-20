# How the price-elasticity model works: overview

This is the plain-language version. It is for readers who are comfortable with numbers but have no
maths background. Each step links to the matching section of the technical document,
[`model_maths.pdf`](model_maths.pdf), where the same step is derived in full. Step N here is
§N there.

**The question:** if we change our mortgage rate by one basis point (bp), how many more or fewer
applications do we get, segment by segment?

---

## 1. Measure our share of the market

The model works segment by segment, for example owner-occupier P&I at LVR up to 80%, and week by
week (or month by month). For each one we know how many loans were lodged across the market and how
many of those were ours. Our share is the simple ratio:

> share = our lodgements ÷ market lodgements

Business plans are made in applications, so we assume applications move in step with our
lodgements. The `conversion_check` output tests this: if the ratio of applications to lodgements
drifts over time, a change in share is not quite a change in applications.

[Technical: §1](model_maths.pdf#section.1)

## 2. Work out the price customers compare us with

A borrower choosing a lender weighs the rates on offer, plus extras such as cashback, which we
convert into an "effective rate". They lean heavily towards the cheapest lenders but don't ignore
the rest. The model turns this into one market benchmark, called **r\***, a "soft cheapest rate".
It sits at or a little below the cheapest competitor. It barely notices expensive lenders. It falls a
little whenever a new competitor appears.

A simple average of competitor rates is the wrong benchmark. It reacts when an uncompetitive lender
reprices, which borrowers barely notice. And it can rise when a new, expensive lender enters, when
in reality borrowers just gained another option.

One consequence: if the number of competitors in our data changes, r\* moves even though nobody
changed a rate. The run warns when that happens. All rates are averaged over the days in each period
so that our rates and competitors' rates are measured the same way.

[Technical: §2](model_maths.pdf#section.2)

## 3. Turn price changes into the price customers *perceive*

Customers and brokers don't notice a price change instantly. The model lets a headline change sink
in gradually: by default half of it is noticed within 2 periods (2 weeks on weekly data). Changes to
discretionary discounts are noticed more slowly (half within 8 periods) and only partly, at 50% by
default, because not every customer sees or gets them. The result is the **perceived price gap**, called x: how much dearer we
look than the market, compared with where we started.

> x = our perceived price change − the market benchmark's perceived change

Without a competitor file, x is just our own price move, with anything common to all segments
stripped out (the `x_basis` column says `move` rather than `gap`). Competitors' moves are then
invisible, so expect noisier estimates.

Competitors' discretionary discounts are invisible to us. That blurs x a little, which makes the
panel estimate (step 4) understate how sensitive customers are.

[Technical: §3](model_maths.pdf#section.3)

## 4. Cross-check with a whole-panel regression

The first estimate compares segments with each other, week by week. Anything that hit every segment
at once cancels out: RBA moves, seasonality, our bank-wide repricing. The model then asks: in the
segments where our relative price rose more, did our share fall more?

This uses all the data, including every competitor move, and competitors' moves are the one source
of price changes that our own motives can't contaminate. But it has a known flaw. When we cut
because share had already slipped, the cut looks less effective than it was, so this estimate
(`beta_fe`) can understate sensitivity, and its error bars are too narrow.

The optional Bayesian version (`beta_hier`, run with `--bayes`) fixes most of that. It lets each
segment's "normal" share drift, so a cut made after share slipped is judged against where share had
slipped to. With competitor data, `beta_hier` is the main estimate and `beta_fe` the cross-check;
if `beta_fe` is well below it, suspect cuts made because share had slipped.

The same step also tests whether the data clearly prefers other speeds of customer response
(step 3). It switches only if the improvement is large (at least 10%).

[Technical: §4](model_maths.pdf#section.4)

## 5. Measure each deliberate repricing directly

For every logged repricing, the model compares the change in our share in that segment, from the
4 periods before to the 8 periods after, with the change in segments we didn't touch over the same
weeks. That comparison removes anything that affected the whole market.

> sensitivity from one event = −(share change vs other segments) ÷ (price change vs other segments)

Every price move is partly strategy, partly a reaction to volume, partly a reaction to
competitors, so the model doesn't ask you to say which. If a campaign, broker incentive or policy
change went out with the price move, list it in the `other_changes` column: the share response
can't then be put down to price, so the event is left out. Setting `clean` to FALSE also keeps an
event out. The remaining events are the **usable** ones (counted in `n_clean_events`). Moves made
because share had slipped will still be in there and make price look a little less effective, which
is one reason these numbers are a check on the rule of thumb rather than the main estimate.

The error bar on each event comes from running the same comparison at quiet times when nothing
changed. That shows how much share wobbles on its own. Events are skipped if they sit too close to
the start or end of the data, if every other segment was also repriced, or if the price barely
moved.

[Technical: §5](model_maths.pdf#section.5)

## 6. Start from the business rule of thumb, then update it segment by segment

The current rule of thumb is that 1bp cheaper brings about 8% more applications. This is converted
into a starting sensitivity for each segment, the **prior**, with a stated level of doubt (±30%).
Each segment's usable events then pull the answer away from the prior. Precise events pull hard,
noisy ones barely at all. The result is `beta_post`, and `prior_weight` says how much of it is still
rule of thumb: 1.0 means all judgment, 0 means all evidence.

The panel regression from step 4 is deliberately **not** added here. It re-uses the same data and
carries the bias described in step 4.

[Technical: §6](model_maths.pdf#section.6)

## 7. Share evidence across segments

If the rule of thumb is wrong, it is probably wrong by a similar factor everywhere. The pooled
estimate (`beta_pool`) uses usable events from *all* segments to estimate that common factor
(`shared_scale`, where 1.0 means the rule of thumb is right). It then lets each segment move away
from the common factor as far as its own events justify (`own_weight`). Segments with no events of
their own still benefit. With no events anywhere, it equals the starting prior.

**What it tells you:** with few usable repricings, these numbers mostly restate the rule of thumb,
so treat them as a test of it rather than the answer (step 4 has the main estimate). If
`shared_scale` is clearly away from 1, the rule of thumb is off everywhere. "Clearly" means its 90%
range, `shared_scale` ± 1.645 × `shared_scale_sd`, excludes 1. If it does, `beta_pool` is the better
of the two judgment-based numbers, and `beta_post` otherwise. In simulations where the rule of
thumb was about 50% too high everywhere, the typical error fell from 61% (`beta_post`) to 38%
(`beta_pool`). Where segments genuinely differed a lot, pooling was slightly worse (30–33% against
28%). `segment_sd_frac` is the judgment setting for how different segments are allowed to be.

[Technical: §7](model_maths.pdf#section.7)

## 8. Track our "natural" share over time

Holding the sensitivity fixed at `beta_post`, the model removes the effect of our pricing from the share
history. What remains is the share we would have had if our price relative to the market had stayed
where it started. This series is smoothed to show the underlying trend (`natural_share`). It still
includes market-wide demand shifts and brand effects, so it is "share before pricing", not "share
due to brand alone".

[Technical: §8](model_maths.pdf#section.8)

## 9. Use it in Excel

Excel takes a proposed price change x (in bp, relative to the market) and returns an applications
multiplier:

> multiplier = 1 ÷ (s0 + (1 − s0) × e^(β × x))

Here s0 is today's share and β is the chosen sensitivity (with competitor data, `beta_hier` is the
main estimate; see step 4). For small changes this is very close to
the rule-of-thumb form: 1bp cheaper gives about u% more applications. For large changes it bends,
because we can never take more than the whole market. With competitor data, x is our change minus the change in
the benchmark r\*, which can be recalculated from the `benchmark` sheet when competitors move. A
change to the discretionary discount counts at the discount-shading factor `kappa` (50% by
default).

The multiplier assumes the market size and conversion rate stay the same. It also assumes today's
share is settled at today's price, so check the `x_drift_s0_window` warning. And it shows the full
effect once customers have caught up, not the first week's effect.

[Technical: §9](model_maths.pdf#section.9)

## 10. Know the limits

- With few usable repricings, `beta_post` and `beta_pool` are mostly the rule of thumb. The outputs
  say how much (`prior_weight`, `shared_scale_sd`). The panel estimates (`beta_hier`, `beta_fe`)
  don't depend on the event log.
- Moves made in every segment at once, such as RBA pass-through, can't be measured as events
  because there is nothing to compare against. A separate RBA cross-check is planned
  ([plan](rba_window_plan.md)) but not built.
- Customers sometimes react when a change is *announced*, before it takes effect. That can make
  share jump around in the weeks in between.
- Competitor discounts we can't see blur the market benchmark.
- Changes in the number of competitors in the data move the benchmark.
- The baseline share must reflect today's price.

[Technical: §10](model_maths.pdf#section.10)
