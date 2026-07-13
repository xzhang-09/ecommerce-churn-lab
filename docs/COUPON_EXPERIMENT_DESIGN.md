# Coupon Experiment Design

## Purpose

This document extends the churn model from risk prediction to a future causal test:
use `churn_probability` to define an at-risk customer pool, randomly assign coupon
treatment inside that pool, and measure whether coupons create incremental
retention or profit.

The current project does **not** include post-coupon outcome data. Therefore this
is an experiment design and pre-analysis plan, not evidence that coupons work.
Any causal claim requires running the randomized experiment and collecting
outcomes after assignment.

## Decision Question

The business question is not simply "who is likely to churn?" It is:

```text
Among customers with high predicted churn risk, does sending a coupon produce
incremental value after accounting for coupon cost?
```

The churn model is used only to target the population and create risk strata. The
effect of the coupon itself must be estimated by random assignment.

## Experiment Population

1. Train the churn model with `churn-pipeline`.
2. Score the candidate customer batch with `churn-score`.
3. Sort customers by `churn_probability` descending.
4. Include the top risk segment in the experiment pool, for example the top 30%.

The top-risk cutoff should be chosen before assignment and should not be changed
after looking at experiment outcomes. A smaller pool concentrates on customers
who are more likely to churn, while a larger pool provides more sample size and
tests whether coupons also work for medium-risk customers.

## Randomization

Use blocked randomization by predicted risk:

1. Split the selected risk pool into ordered risk bands, such as high, medium,
   and lower-within-pool risk.
2. Within each band, randomly assign customers to:
   - `treatment`: receive the coupon.
   - `control`: no coupon, or the current standard retention process.
3. Use a fixed random seed so the assignment is reproducible.

Blocked randomization keeps treatment and control groups balanced by model risk.
This matters because customers at the very top of the score distribution may
have different baseline churn rates than customers near the cutoff.

The helper script creates the assignment table from a scored customer file:

```bash
python scripts/assign_coupon_experiment.py \
  --input data/processed/churn_scores.csv \
  --output data/processed/coupon_experiment_assignments.csv \
  --risk-pool-fraction 0.30 \
  --treatment-fraction 0.50 \
  --risk-bands 3 \
  --random-state 42
```

The output has one row per experiment customer:

```text
CustomerID, churn_probability, risk_rank, risk_band, treatment_group
```

The script validates the scored input before assigning anything: customer ids must
be unique and every row needs a probability in [0, 1]. Because assignment is a
one-shot operation, it also writes a `*_metadata.json` audit file next to the CSV
(parameters, seed, input row count and SHA-256, timestamp, group sizes) and prints
a per-band treatment/control balance check (group sizes and mean probability).

Edge case: if a band contains a single customer, that customer is assigned by a
coin flip weighted by the treatment fraction — not forced into treatment — so tiny
bands do not systematically inflate the treatment group. Such a band has no
internal control, so prefer fewer `--risk-bands` when the pool is small.

## Outcomes

Define the outcome window before launch, for example 30 or 60 days after coupon
assignment. The final choice should match the normal purchase cadence.

Primary outcome:

- Repeat purchase indicator during the outcome window, or churn indicator by the
  end of the outcome window.

Business outcome:

- Incremental gross margin net of coupon cost.

Guardrail outcomes:

- Coupon redemption cost.
- Return or cancellation rate.
- Customer complaints.
- Excess discounting of customers who would have purchased anyway.

## Analysis Plan

The primary analysis should be intent-to-treat: compare everyone assigned to
treatment with everyone assigned to control, regardless of whether a customer
actually redeems the coupon. This preserves the randomization.

For a binary outcome such as repeat purchase:

```text
ATE = mean(outcome | treatment) - mean(outcome | control)
```

Report:

- Treatment and control sample sizes.
- Treatment and control outcome rates.
- Difference in rates.
- Confidence interval.
- p-value or randomization-test result.

This primary analysis is implemented ahead of time in
`churn-analyze-coupon-experiment` (`scripts/analyze_coupon_experiment.py`), so it is
locked in before any outcomes exist and cannot be tailored to results. The script
takes the assignment file plus an outcomes CSV (one row per assigned customer with a
binary `outcome` column), reports the rate difference with a Wald confidence
interval and a randomization-test p-value (labels shuffled within risk bands,
mirroring the blocked assignment), and refuses to run if any assigned customer
lacks an outcome — dropping unmatched customers would undo the randomization, so
the outcome must be defined to exist for everyone (e.g. churned by end of window).

For profit:

```text
incremental_profit =
    incremental_gross_margin
    - coupon_face_value_or_discount_cost
    - delivery_or_campaign_cost
```

The experiment is successful only if the effect is positive on the business
metric, not merely if churn decreases.

## Risk-Band Reporting

In addition to the overall effect, report effects by `risk_band`. This answers a
practical targeting question:

```text
Do coupons work best for the highest-risk customers, or only for customers who
are still persuadable?
```

Band-level analysis should be treated as secondary unless the experiment is
powered for it. Small bands can produce noisy estimates.

## Sample Size

Without historical coupon experiments, sample size depends on assumptions. Before
launch, choose:

- Baseline control outcome rate.
- Minimum detectable effect worth acting on.
- Significance level, commonly 5%.
- Power, commonly 80%.

Example: if the control repeat-purchase rate is expected to be 20%, the team
should decide whether a 2, 3, or 5 percentage-point lift is the minimum business
effect worth detecting. Smaller detectable effects require much larger samples.

If the available high-risk population is too small, prefer a wider risk pool or a
longer experiment window rather than changing the analysis after outcomes arrive.

## Limitations

- The churn model identifies risk, not coupon responsiveness.
- High churn probability does not imply high incremental coupon effect.
- This project currently has no post-assignment outcomes, so it cannot estimate
  real uplift or ROI yet.
- Any analysis using historical non-random coupon sends would be observational
  and vulnerable to selection bias.

## Recommended Next Step

Use this project to produce the scored list and the randomized assignment file.
After the campaign window closes, collect a binary outcome for every assigned
customer and run the pre-specified intent-to-treat analysis with
`churn-analyze-coupon-experiment`. The profit analysis (incremental gross margin
net of coupon cost) still has to be done separately once real cost inputs exist —
the script deliberately covers only the binary primary outcome.
