import json

import pandas as pd
import pytest

from churn.experiments.coupon_assignment import (
    assign_coupon_experiment,
    main,
    summarize_balance,
)


def test_assign_coupon_experiment_selects_top_risk_and_balances_within_bands():
    scored = pd.DataFrame(
        {
            "CustomerID": range(1, 11),
            "churn_probability": [0.91, 0.88, 0.81, 0.74, 0.69, 0.61, 0.52, 0.44, 0.33, 0.20],
        }
    )

    assigned = assign_coupon_experiment(
        scored,
        risk_pool_fraction=0.6,
        treatment_fraction=0.5,
        n_risk_bands=3,
        random_state=7,
    )

    assert len(assigned) == 6
    assert set(assigned["CustomerID"]) == {1, 2, 3, 4, 5, 6}
    assert list(assigned.columns) == [
        "CustomerID",
        "churn_probability",
        "risk_rank",
        "risk_band",
        "treatment_group",
    ]
    assert assigned["risk_rank"].tolist() == [1, 2, 3, 4, 5, 6]
    assert set(assigned["risk_band"]) == {"risk_band_1_highest", "risk_band_2", "risk_band_3"}

    by_band = assigned.groupby("risk_band")["treatment_group"].value_counts().unstack(fill_value=0)
    assert (by_band["treatment"] == 1).all()
    assert (by_band["control"] == 1).all()


def test_assign_coupon_experiment_is_reproducible_for_same_seed():
    scored = pd.DataFrame(
        {
            "CustomerID": range(1, 13),
            "churn_probability": [0.95, 0.90, 0.84, 0.79, 0.72, 0.68, 0.62, 0.55, 0.49, 0.41, 0.30, 0.22],
        }
    )

    first = assign_coupon_experiment(scored, risk_pool_fraction=0.75, random_state=42)
    second = assign_coupon_experiment(scored, risk_pool_fraction=0.75, random_state=42)

    pd.testing.assert_frame_equal(first, second)


def test_assign_coupon_experiment_rejects_invalid_fractions():
    scored = pd.DataFrame({"CustomerID": [1], "churn_probability": [0.9]})

    with pytest.raises(ValueError, match="risk_pool_fraction"):
        assign_coupon_experiment(scored, risk_pool_fraction=0)
    with pytest.raises(ValueError, match="treatment_fraction"):
        assign_coupon_experiment(scored, treatment_fraction=1.0)


def test_assign_coupon_experiment_rejects_bad_customer_data():
    duplicated_ids = pd.DataFrame(
        {"CustomerID": [1, 1, 2], "churn_probability": [0.9, 0.8, 0.7]}
    )
    with pytest.raises(ValueError, match="Duplicate ids"):
        assign_coupon_experiment(duplicated_ids)

    missing_probability = pd.DataFrame(
        {"CustomerID": [1, 2], "churn_probability": [0.9, None]}
    )
    with pytest.raises(ValueError, match="missing"):
        assign_coupon_experiment(missing_probability)

    out_of_range = pd.DataFrame({"CustomerID": [1, 2], "churn_probability": [0.9, 1.2]})
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        assign_coupon_experiment(out_of_range)


def test_single_customer_band_is_randomized_not_forced_to_treatment():
    # One customer, one band: the assignment must be a weighted coin flip, not a
    # guaranteed treatment. Across seeds both groups have to show up, and each
    # seed's result must be deterministic.
    scored = pd.DataFrame({"CustomerID": [1], "churn_probability": [0.9]})

    groups = set()
    for seed in range(20):
        assigned = assign_coupon_experiment(
            scored, risk_pool_fraction=1.0, n_risk_bands=1, random_state=seed
        )
        repeat = assign_coupon_experiment(
            scored, risk_pool_fraction=1.0, n_risk_bands=1, random_state=seed
        )
        pd.testing.assert_frame_equal(assigned, repeat)
        groups.add(assigned["treatment_group"].iloc[0])

    assert groups == {"treatment", "control"}


def test_main_writes_assignments_metadata_and_balance(tmp_path, capsys):
    scored = pd.DataFrame(
        {
            "CustomerID": range(1, 11),
            "churn_probability": [0.91, 0.88, 0.81, 0.74, 0.69, 0.61, 0.52, 0.44, 0.33, 0.20],
        }
    )
    input_path = tmp_path / "churn_scores.csv"
    output_path = tmp_path / "assignments.csv"
    scored.to_csv(input_path, index=False)

    main(
        [
            "--input", str(input_path),
            "--output", str(output_path),
            "--risk-pool-fraction", "0.6",
            "--risk-bands", "3",
            "--random-state", "7",
        ]
    )

    assigned = pd.read_csv(output_path)
    assert len(assigned) == 6

    metadata = json.loads((tmp_path / "assignments_metadata.json").read_text())
    assert metadata["input"]["rows"] == 10
    assert len(metadata["input"]["sha256"]) == 64
    assert metadata["parameters"]["random_state"] == 7
    assert metadata["parameters"]["risk_pool_fraction"] == 0.6
    assert metadata["output"]["rows"] == 6
    assert metadata["output"]["n_treatment"] + metadata["output"]["n_control"] == 6

    balance = summarize_balance(assigned)
    assert set(balance.columns) == {
        "risk_band", "treatment_group", "n_customers", "mean_probability"
    }
    assert balance["n_customers"].sum() == 6
    assert "Balance check" in capsys.readouterr().out
