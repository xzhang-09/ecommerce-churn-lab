#!/usr/bin/env python3
"""Pre-specified intent-to-treat analysis for the coupon experiment.

This module locks in the primary analysis from docs/COUPON_EXPERIMENT_DESIGN.md
before any outcome data exists: compare everyone assigned to treatment with
everyone assigned to control, regardless of coupon redemption. It covers the
binary primary outcome only — profit analysis needs cost inputs that do not
exist yet and is intentionally out of scope.
"""

import argparse
import json
import math
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import norm

from churn.experiments.coupon_assignment import _sha256_of_file

DEFAULT_ID_COLUMN = "CustomerID"
DEFAULT_OUTCOME_COLUMN = "outcome"
GROUP_COLUMN = "treatment_group"
BAND_COLUMN = "risk_band"


def _as_float_or_none(value) -> float | None:
    value = float(value)
    return None if math.isnan(value) else value


def _rate(outcomes: np.ndarray) -> float:
    return float(outcomes.mean()) if len(outcomes) else float("nan")


def _merge_validated(
    assignments: pd.DataFrame,
    outcomes: pd.DataFrame,
    id_column: str,
    outcome_column: str,
) -> pd.DataFrame:
    for name, frame, required in (
        ("assignments", assignments, [id_column, GROUP_COLUMN, BAND_COLUMN]),
        ("outcomes", outcomes, [id_column, outcome_column]),
    ):
        missing_columns = [column for column in required if column not in frame.columns]
        if missing_columns:
            raise ValueError(f"{name} input is missing columns: {missing_columns}")
        if frame[id_column].duplicated().any():
            raise ValueError(f"{name} input has duplicate ids in '{id_column}'")
    if assignments.empty:
        raise ValueError("assignments input is empty; nothing to analyze")

    unexpected_groups = set(assignments[GROUP_COLUMN]) - {"treatment", "control"}
    if unexpected_groups:
        raise ValueError(f"Unexpected treatment_group values: {sorted(unexpected_groups)}")

    merged = assignments.merge(
        outcomes[[id_column, outcome_column]], on=id_column, how="left"
    )
    # Intent-to-treat needs an outcome for every assigned customer. Silently
    # dropping unmatched rows would undo the randomization, so fail instead.
    n_missing = int(merged[outcome_column].isna().sum())
    if n_missing:
        raise ValueError(
            f"{n_missing} assigned customers have no outcome row; intent-to-treat "
            "requires an outcome for everyone assigned (define the outcome so it "
            "exists for every customer, e.g. churned-by-end-of-window)"
        )
    observed_values = set(pd.unique(merged[outcome_column]))
    if not observed_values <= {0, 1}:
        raise ValueError(
            f"'{outcome_column}' must be binary 0/1; found values {sorted(observed_values)[:5]}"
        )
    merged[outcome_column] = merged[outcome_column].astype(int)
    if not (merged[GROUP_COLUMN] == "treatment").any() or not (
        merged[GROUP_COLUMN] == "control"
    ).any():
        raise ValueError("Both treatment and control customers are required")
    return merged


def _permutation_p_value(
    merged: pd.DataFrame,
    outcome_column: str,
    observed_difference: float,
    n_permutations: int,
    random_state: int,
) -> float:
    """Two-sided randomization test, shuffling labels within each risk band.

    The assignment was randomized within bands, so the reference distribution
    must be generated the same way.
    """
    rng = np.random.default_rng(random_state)
    labels = merged[GROUP_COLUMN].to_numpy()
    outcome_values = merged[outcome_column].to_numpy()
    band_positions = [
        np.where(merged[BAND_COLUMN].to_numpy() == band)[0]
        for band in merged[BAND_COLUMN].unique()
    ]

    n_at_least_as_extreme = 0
    for _ in range(n_permutations):
        permuted = labels.copy()
        for positions in band_positions:
            permuted[positions] = rng.permutation(permuted[positions])
        difference = _rate(outcome_values[permuted == "treatment"]) - _rate(
            outcome_values[permuted == "control"]
        )
        if abs(difference) >= abs(observed_difference) - 1e-12:
            n_at_least_as_extreme += 1
    return (n_at_least_as_extreme + 1) / (n_permutations + 1)


def analyze_coupon_experiment(
    assignments: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    id_column: str = DEFAULT_ID_COLUMN,
    outcome_column: str = DEFAULT_OUTCOME_COLUMN,
    alpha: float = 0.05,
    n_permutations: int = 10_000,
    random_state: int = 42,
) -> dict:
    """Run the pre-specified intent-to-treat analysis on collected outcomes.

    ``assignments`` is the output of ``churn-assign-coupon-experiment``;
    ``outcomes`` has one row per assigned customer with a binary outcome.
    Returns a report dict with the overall ITT estimate (difference in outcome
    rates, Wald confidence interval, within-band permutation p-value) and a
    secondary per-band breakdown.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    if n_permutations < 1:
        raise ValueError(f"n_permutations must be >= 1; got {n_permutations}")
    merged = _merge_validated(assignments, outcomes, id_column, outcome_column)

    treated = merged.loc[merged[GROUP_COLUMN] == "treatment", outcome_column].to_numpy()
    control = merged.loc[merged[GROUP_COLUMN] == "control", outcome_column].to_numpy()
    rate_treatment, rate_control = _rate(treated), _rate(control)
    difference = rate_treatment - rate_control

    z = float(norm.ppf(1 - alpha / 2))
    standard_error = math.sqrt(
        rate_treatment * (1 - rate_treatment) / len(treated)
        + rate_control * (1 - rate_control) / len(control)
    )
    p_value = _permutation_p_value(
        merged, outcome_column, difference, n_permutations, random_state
    )

    by_band = []
    for band, band_frame in merged.groupby(BAND_COLUMN, sort=False):
        band_treated = band_frame.loc[
            band_frame[GROUP_COLUMN] == "treatment", outcome_column
        ].to_numpy()
        band_control = band_frame.loc[
            band_frame[GROUP_COLUMN] == "control", outcome_column
        ].to_numpy()
        band_rate_treatment, band_rate_control = _rate(band_treated), _rate(band_control)
        by_band.append(
            {
                "risk_band": band,
                "n_treatment": len(band_treated),
                "n_control": len(band_control),
                "rate_treatment": _as_float_or_none(band_rate_treatment),
                "rate_control": _as_float_or_none(band_rate_control),
                "difference": _as_float_or_none(band_rate_treatment - band_rate_control),
            }
        )

    return {
        "overall": {
            "n_treatment": len(treated),
            "n_control": len(control),
            "rate_treatment": rate_treatment,
            "rate_control": rate_control,
            "difference": difference,
            "ci_low": difference - z * standard_error,
            "ci_high": difference + z * standard_error,
            "alpha": alpha,
            "p_value_permutation": p_value,
            "n_permutations": n_permutations,
        },
        "by_band": by_band,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Pre-specified intent-to-treat analysis of the coupon experiment."
    )
    parser.add_argument("--assignments", required=True,
                        help="CSV from churn-assign-coupon-experiment")
    parser.add_argument("--outcomes", required=True,
                        help="CSV with one row per assigned customer and a binary outcome")
    parser.add_argument("--output", default="data/processed/coupon_experiment_analysis.json",
                        help="where to write the JSON analysis report")
    parser.add_argument("--id-column", default=DEFAULT_ID_COLUMN,
                        help="customer id column shared by both inputs")
    parser.add_argument("--outcome-column", default=DEFAULT_OUTCOME_COLUMN,
                        help="binary outcome column in the outcomes file")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="significance level for the confidence interval")
    parser.add_argument("--permutations", type=int, default=10_000,
                        help="iterations for the within-band randomization test")
    parser.add_argument("--random-state", type=int, default=42,
                        help="random seed for the randomization test")
    args = parser.parse_args(argv)

    assignments = pd.read_csv(args.assignments)
    outcomes = pd.read_csv(args.outcomes)
    report = analyze_coupon_experiment(
        assignments,
        outcomes,
        id_column=args.id_column,
        outcome_column=args.outcome_column,
        alpha=args.alpha,
        n_permutations=args.permutations,
        random_state=args.random_state,
    )
    report["metadata"] = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "assignments": {"path": args.assignments, "sha256": _sha256_of_file(args.assignments)},
        "outcomes": {"path": args.outcomes, "sha256": _sha256_of_file(args.outcomes)},
        "outcome_column": args.outcome_column,
        "random_state": args.random_state,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    overall = report["overall"]
    confidence = round((1 - overall["alpha"]) * 100)
    print("Intent-to-treat analysis (primary):")
    print(
        f"  treatment: {overall['n_treatment']} customers, "
        f"outcome rate {overall['rate_treatment']:.4f}"
    )
    print(
        f"  control:   {overall['n_control']} customers, "
        f"outcome rate {overall['rate_control']:.4f}"
    )
    print(
        f"  difference: {overall['difference']:+.4f} "
        f"({confidence}% CI {overall['ci_low']:+.4f} to {overall['ci_high']:+.4f}), "
        f"permutation p-value {overall['p_value_permutation']:.4f}"
    )
    print("Band-level effects (secondary, exploratory unless powered):")
    print(pd.DataFrame(report["by_band"]).to_string(index=False))
    print(f"Wrote analysis report to {args.output}.")
    print(
        "Reminder: the business success criterion is positive incremental profit "
        "net of coupon cost, not outcome lift alone (docs/COUPON_EXPERIMENT_DESIGN.md)."
    )


if __name__ == "__main__":
    main()
