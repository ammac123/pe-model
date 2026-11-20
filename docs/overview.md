# How the price-elasticity model works: overview

This is the plain-language version. It is for readers who are comfortable with numbers but have no
maths background. Each step links to the matching section of the technical document,
[`model_maths.pdf`](model_maths.pdf), where the same step is derived in full. Step N here is
§N there.

**The question:** if we change our mortgage rate in a segment by some number of basis points (bp),
what happens to our applications and share over the next 12 weeks, and how sure are we?

**The answer comes from one model.** It produces one sensitivity per segment, always with an 80%
range: four times in five the truth should lie inside it, one time in five outside.

---

## 1. Measure our share of the market

The model works segment by segment (purpose × repayment type × LVR band, for example
owner-occupier P&I at 70–80% LVR) and week by week. For each one we know how many loans were lodged
across the broker market and how many of those were ours.

> share = our lodgements ÷ market lodgements

The model works with the counts themselves, not just the ratio. A week with 3 lodgements out of 40
says much less than a week with 300 out of 4,000, and the model weighs it accordingly. Weeks with
no lodgements of ours are used like any other.

Business plans are made in applications, so we assume applications move in step with our
lodgements. The `conversion` columns in `segments.csv` test this.

[Technical: §1](model_maths.pdf#section.1)

## 2. Work out the price customers compare us with

A borrower choosing a lender weighs the rates on offer, plus extras such as cashback, which you
enter by hand as an adjustment. They lean heavily towards the cheapest lenders but don't ignore
the rest. The model turns this into one market benchmark, called **r\***, a "soft cheapest rate".
It sits at or a little below the cheapest competitor, barely notices expensive lenders, and falls
a little whenever a new competitor appears.

How "soft" it is depends on how price-sensitive customers are, which is the thing being estimated.
So the model solves the two together: it estimates sensitivity, rebuilds the benchmark to match,
and repeats until they agree. The run log shows each round and warns loudly if they never settle.

Carded rates are a weak guide to the rates people pay. Your manual adjustments are the only way
the model can see competitor discounting, so keep a dated record of them.

[Technical: §2](model_maths.pdf#section.2)

## 3. Turn price changes into the price gap customers feel

The **price gap** is how much dearer we look than the market, compared with where we started:

> gap = our headline change + 50% of our discretionary discount change − the benchmark's change

Discount changes count at 50% by default, because not every customer sees or gets them.

Customers and brokers don't react instantly. The model lets a change in the gap sink in gradually,
with one **half-life**: the number of weeks until half the change is felt. It starts from a guess
of two weeks and learns the figure from the data. At a two-week half-life, only about 80% of a
move is felt on average over its first 12 weeks. That is why the forecast is a 12-week path and not
a single long-run figure.

[Technical: §3](model_maths.pdf#section.3)

## 4. Separate price from everything else: natural share

Share moves for many reasons besides price: turnaround time, credit policy, service, brand, the
season. The model collects all of them into each segment's **natural share**, the share we would
have had if our price gap had stayed where it started. Natural share is allowed to drift from week
to week, in two parts:

- a smooth **bank-wide trend**, common to every segment, and
- a smooth **segment trend** for whatever is specific to that segment.

When you log a non-price change, such as a campaign or a policy change, natural share is also
allowed one sudden jump in the affected segments that week. A price move that lands in the same
week as a logged change then teaches the model little. That is intended: the two cannot be told
apart.

Because natural share drifts, a price cut made after share had already slipped is judged against
where share had slipped to, not against its long-run average. That removes the main bias in
simpler methods.

One assumption matters here. Many of our price moves are bank-wide. The model can learn from them
only because the bank-wide trend is assumed to be smooth. So an abrupt bank-wide shock in the same
week as a bank-wide price move would be read as price response, unless you log it.

[Technical: §4](model_maths.pdf#section.4)

## 5. Start from a documented range for sensitivity, and share evidence across segments

Sensitivity is expressed as **uplift per bp**: the percentage rise in our volume from a 1bp cut.

The model does not start from a single figure. It starts from a range: middle 3% per bp, with an
80% range of 1.2% to 7.4%. Published studies of other lenders sit at 1 to 2.5%. The range is set
above that because we are a small, price-focused lender, and wide enough that the old 8% rule of
thumb is still possible (about a 1-in-12 chance at the start) but is not assumed. The data then
narrow the range.

With about 10 price moves across 20 segments, most segments have little evidence of their own. So
the model lets segments inform each other: investor segments share a common difference from
owner-occupier ones, interest-only from P&I, and each LVR band has its own difference. A segment
with no price move of its own still gets a figure from the segments it resembles.

Each segment's output carries `prior_share`: how much of the answer is still the starting range.
Near 0, the data decided it. Near 1, the number is mostly judgment.

[Technical: §5](model_maths.pdf#section.5)

## 6. Fit everything at once

All of the above is one model, fitted in one go: natural share, the half-life and every segment's
sensitivity are estimated together, so each is judged with the others' uncertainty taken into
account. The fitting method produces thousands of equally plausible versions of the whole model.
Every reported range is read off those versions: the 10th, 50th and 90th percentile.

The run checks that the fit is numerically sound and writes `UNRELIABLE RUN` in the log if it is
not. It also prints the starting range it used, so you can confirm it is the one you intended.

[Technical: §6](model_maths.pdf#section.6)

## 7. Forecast a price change

For a proposed change, the model plays each plausible version of itself forward for 12 weeks,
with and without the change, and compares the two. It reports the 12-week applications multiplier,
the 12-week share and the weekly path, each with its 80% range. If a recent move has not finished
sinking in, the "without" case carries it on.

Two ranges are reported, and they answer different questions:

- **Price-effect range.** Natural share stays where it is today. This is the "no competitor
  response and no market movement" condition. It reflects how well we know the sensitivity, the
  half-life and today's level.
- **Full predictive range.** Natural share also keeps drifting and weekly counts are noisy. This
  is what to expect when you later compare a forecast with what happened.

The long-run effect, once customers have fully caught up, is shown separately. It is larger than
the 12-week effect and is not the 12-week answer.

The Excel workbook does the price-effect calculation with live formulas, for the low, middle and
high sensitivity:

> weekly multiplier = 1 ÷ (s + (1 − s) × e^(β × gap change × share of the change felt by that week))

where s is today's share, and the 12-week multiplier is the average of the weekly ones. Competitor
scenarios are typed into the `benchmark` sheet.

[Technical: §7](model_maths.pdf#section.7)

## 8. Know how accurate it can be, and check it

**The accuracy table** (`accuracy.csv`) states, for every segment and group and for moves of ±5,
±10 and ±25bp, how wide the 80% range of the 12-week forecast is, and whether it is inside ±5% of
applications and ±0.3 points of share. Three things add up:

- how well we know the sensitivity, which matters more the bigger the move;
- chance variation in weekly counts, which is large in thin segments (below about 50 of our
  lodgements a week it alone exceeds ±5%);
- drift in natural share, which no pricing model controls.

Expect the target to be met for small moves in large segments and in groups, and not for 25bp
moves or thin segments. Nothing is tuned to make the table pass.

**Checks that need no model.** `check-moves` shows applications before and after each past price
move, beside what 8% and 3% per bp would imply. Under 8% a 10bp cut should roughly double
applications. Look at this first.

**Checks on the model**, each a rerun:

- *Leave one move out.* Hide the 12 weeks after a past move, refit, and see whether the 80% range
  contains what happened.
- *Quiet periods.* Forecast 12 weeks ahead for segments we did not reprice. This shows the error
  that remains even when price does nothing.
- *Placebo.* Invent a 10bp move at a quiet date. The model should find no effect about four times
  in five.
- *Sensitivity runs.* Refit with the starting range centred on 2, 4 and 8%, with a softer and a
  harder benchmark, with discounts counted at 30% and 70%, and with two structural alternatives.
  If the answer moves a lot, it depends on that judgment.

Two older estimators (a whole-panel regression and before-and-after comparisons of each move) are
reported beside the model's figures as a cross-check. Both are known to understate sensitivity
when we cut price after share has slipped, and neither can use bank-wide moves. They do not feed
the estimate.

[Technical: §8](model_maths.pdf#section.8)

## 9. Know the limits

- The data hold about 10 price moves. Where `prior_share` is high, the figure is mostly judgment.
- An unlogged bank-wide shock in the same week as a bank-wide price move is read as price response.
- Competitor discounts we cannot see blur the benchmark and make customers look less
  price-sensitive than they are.
- Turnaround time is inside natural share. If a price cut causes a surge that slows turnaround,
  the estimate is the net effect of both: right for forecasting if capacity behaves the same way
  next time, too low if capacity has since improved.
- "No competitor response and no market movement" is never observed. It can only be tested by
  backtests that feed in the competitor rates and market volumes that occurred.
- Customers sometimes react when a change is *announced*, before it takes effect.
- Changes in the number of competitors in the data move the benchmark.

[Technical: §9](model_maths.pdf#section.9)
