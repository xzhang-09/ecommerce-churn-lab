#!/usr/bin/env python3
"""Assign high-risk customers to a coupon randomized experiment."""

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DEFAULT_ID_COLUMN = "CustomerID"
PROBABILITY_COLUMN = "churn_probability"


def _validate_fraction(name: str, value: float, *, upper_inclusive: bool) -> None:
    upper_ok = value <= 1 if upper_inclusive else value < 1
    if value <= 0 or not upper_ok:
        bound = "(0, 1]" if upper_inclusive else "(0, 1)"
        raise ValueError(f"{name} must be in {bound}; got {value}")


def _risk_band_labels(n_rows: int, n_bands: int) -> list[str]:
    labels: list[str] = []
    bands = min(n_bands, n_rows)
    for band_idx, indices in enumerate(np.array_split(np.arange(n_rows), bands), start=1):
        label = "risk_band_1_highest" if band_idx == 1 else f"risk_band_{band_idx}"
        labels.extend([label] * len(indices))
    return labels


def assign_coupon_experiment(
    scored: pd.DataFrame,
    *,
    id_column: str = DEFAULT_ID_COLUMN,
    probability_column: str = PROBABILITY_COLUMN,
    risk_pool_fraction: float = 0.30,
    treatment_fraction: float = 0.50,
    n_risk_bands: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create a stratified treatment/control assignment from scored customers.

    The input is the output of ``churn-score``: one row per customer with a churn
    probability. Only the highest-risk fraction enters the experiment pool. That
    pool is split into risk bands by rank, then customers are randomized within
    each band so treatment/control have similar risk distributions.
    """
    if id_column not in scored.columns:
        raise ValueError(f"Input must include id column '{id_column}'")
    if probability_column not in scored.columns:
        raise ValueError(f"Input must include probability column '{probability_column}'")
    duplicated_ids = scored[id_column][scored[id_column].duplicated()].unique()
    if len(duplicated_ids) > 0:
        raise ValueError(
            f"Duplicate ids in '{id_column}' could place the same customer in both "
            f"experiment groups; first duplicates: {duplicated_ids[:5].tolist()}"
        )
    probabilities = scored[probability_column]
    if probabilities.isna().any():
        raise ValueError(
            f"'{probability_column}' contains {int(probabilities.isna().sum())} missing "
            "values; scored input must have a probability for every customer"
        )
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(f"'{probability_column}' must contain probabilities in [0, 1]")
    _validate_fraction("risk_pool_fraction", risk_pool_fraction, upper_inclusive=True)
    _validate_fraction("treatment_fraction", treatment_fraction, upper_inclusive=False)
    if n_risk_bands < 1:
        raise ValueError(f"n_risk_bands must be >= 1; got {n_risk_bands}")
    if scored.empty:
        return pd.DataFrame(
            columns=[id_column, probability_column, "risk_rank", "risk_band", "treatment_group"]
        )

    ranked = (
        scored[[id_column, probability_column]]
        .copy()
        .sort_values(probability_column, ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )
    pool_size = max(1, math.ceil(len(ranked) * risk_pool_fraction))
    pool = ranked.head(pool_size).copy()
    pool["risk_rank"] = np.arange(1, len(pool) + 1)
    pool["risk_band"] = _risk_band_labels(len(pool), n_risk_bands)
    pool["treatment_group"] = "control"

    rng = np.random.default_rng(random_state)
    for _, band_idx in pool.groupby("risk_band", sort=False).groups.items():
        band_positions = np.array(list(band_idx))
        n_treat = int(round(len(band_positions) * treatment_fraction))
        if len(band_positions) > 1:
            n_treat = min(max(1, n_treat), len(band_positions) - 1)
        else:
            # A one-customer band cannot hold both groups. A weighted coin flip keeps
            # the assignment random instead of systematically pushing these customers
            # into treatment.
            n_treat = int(rng.random() < treatment_fraction)
        treated = rng.choice(band_positions, size=n_treat, replace=False)
        pool.loc[treated, "treatment_group"] = "treatment"

    return pool[[id_column, probability_column, "risk_rank", "risk_band", "treatment_group"]]


def summarize_balance(
    assigned: pd.DataFrame, probability_column: str = PROBABILITY_COLUMN
) -> pd.DataFrame:
    """Per-band group sizes and mean risk, to eyeball treatment/control balance."""
    return (
        assigned.groupby(["risk_band", "treatment_group"], sort=False)[probability_column]
        .agg(n_customers="size", mean_probability="mean")
        .reset_index()
    )


def _sha256_of_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Assign scored customers to a stratified coupon experiment."
    )
    parser.add_argument("--input", required=True,
                        help="CSV from churn-score with customer id and churn_probability")
    parser.add_argument("--output", default="data/processed/coupon_experiment_assignments.csv",
                        help="where to write the assignment CSV")
    parser.add_argument("--id-column", default=DEFAULT_ID_COLUMN,
                        help="customer id column in the scored input")
    parser.add_argument("--probability-column", default=PROBABILITY_COLUMN,
                        help="churn probability column in the scored input")
    parser.add_argument("--risk-pool-fraction", type=float, default=0.30,
                        help="top risk fraction included in the experiment pool")
    parser.add_argument("--treatment-fraction", type=float, default=0.50,
                        help="fraction randomized to coupon treatment within each risk band")
    parser.add_argument("--risk-bands", type=int, default=3,
                        help="number of risk strata for blocked randomization")
    parser.add_argument("--random-state", type=int, default=42,
                        help="random seed for reproducible assignment")
    args = parser.parse_args(argv)

    scored = pd.read_csv(args.input)
    assigned = assign_coupon_experiment(
        scored,
        id_column=args.id_column,
        probability_column=args.probability_column,
        risk_pool_fraction=args.risk_pool_fraction,
        treatment_fraction=args.treatment_fraction,
        n_risk_bands=args.risk_bands,
        random_state=args.random_state,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    assigned.to_csv(args.output, index=False)
    n_treatment = int((assigned["treatment_group"] == "treatment").sum())
    n_control = int((assigned["treatment_group"] == "control").sum())

    # Experiment assignment is a one-shot operation, so record everything needed to
    # audit it later: parameters, seed, and a fingerprint of the exact input scored.
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input": {
            "path": args.input,
            "rows": int(len(scored)),
            "sha256": _sha256_of_file(args.input),
        },
        "parameters": {
            "id_column": args.id_column,
            "probability_column": args.probability_column,
            "risk_pool_fraction": args.risk_pool_fraction,
            "treatment_fraction": args.treatment_fraction,
            "risk_bands": args.risk_bands,
            "random_state": args.random_state,
        },
        "output": {
            "path": args.output,
            "rows": int(len(assigned)),
            "n_treatment": n_treatment,
            "n_control": n_control,
        },
    }
    metadata_path = os.path.splitext(args.output)[0] + "_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(
        f"Wrote {len(assigned)} experiment assignments to {args.output} "
        f"({n_treatment} treatment, {n_control} control)."
    )
    print(f"Wrote assignment metadata to {metadata_path}.")
    if not assigned.empty:
        print("Balance check (per risk band):")
        print(summarize_balance(assigned, args.probability_column).to_string(index=False))


if __name__ == "__main__":
    main()

