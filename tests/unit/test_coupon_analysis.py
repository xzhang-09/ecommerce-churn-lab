import json

import pandas as pd
import pytest

from churn.experiments.coupon_analysis import analyze_coupon_experiment, main


def _make_experiment(n_per_arm_per_band=50, treatment_rate=0.5, control_rate=0.2):
    """Deterministic two-band experiment with exact per-arm outcome rates."""
    rows = []
    customer_id = 1
    for band in ["risk_band_1_highest", "risk_band_2"]:
        for group, rate in [("treatment", treatment_rate), ("control", control_rate)]:
            n_positive = int(round(n_per_arm_per_band * rate))
            for i in range(n_per_arm_per_band):
                rows.append(
                    {
                        "CustomerID": customer_id,
                        "risk_band": band,
                        "treatment_group": group,
                        "outcome": int(i < n_positive),
                    }
                )
                customer_id += 1
    frame = pd.DataFrame(rows)
    assignments = frame[["CustomerID", "risk_band", "treatment_group"]]
    outcomes = frame[["CustomerID", "outcome"]]
    return assignments, outcomes


def test_detects_a_real_lift():
    assignments, outcomes = _make_experiment(treatment_rate=0.5, control_rate=0.2)

    report = analyze_coupon_experiment(
        assignments, outcomes, n_permutations=500, random_state=0
    )

    overall = report["overall"]
    assert overall["n_treatment"] == 100
    assert overall["n_control"] == 100
    assert overall["rate_treatment"] == pytest.approx(0.5)
    assert overall["rate_control"] == pytest.approx(0.2)
    assert overall["difference"] == pytest.approx(0.3)
    assert overall["ci_low"] > 0
    assert overall["p_value_permutation"] < 0.05

    assert [band["risk_band"] for band in report["by_band"]] == [
        "risk_band_1_highest",
        "risk_band_2",
    ]
    for band in report["by_band"]:
        assert band["n_treatment"] == 50
        assert band["n_control"] == 50
        assert band["difference"] == pytest.approx(0.3)


def test_null_effect_is_not_declared_significant():
    assignments, outcomes = _make_experiment(treatment_rate=0.3, control_rate=0.3)

    report = analyze_coupon_experiment(
        assignments, outcomes, n_permutations=500, random_state=0
    )

    assert report["overall"]["difference"] == pytest.approx(0.0)
    assert report["overall"]["p_value_permutation"] == pytest.approx(1.0)


def test_analysis_is_reproducible_for_same_seed():
    assignments, outcomes = _make_experiment(n_per_arm_per_band=10)

    first = analyze_coupon_experiment(assignments, outcomes, n_permutations=200, random_state=7)
    second = analyze_coupon_experiment(assignments, outcomes, n_permutations=200, random_state=7)

    assert first == second


def test_rejects_missing_outcomes_for_assigned_customers():
    assignments, outcomes = _make_experiment(n_per_arm_per_band=5)
    incomplete = outcomes.iloc[:-3]

    with pytest.raises(ValueError, match="no outcome row"):
        analyze_coupon_experiment(assignments, incomplete)


def test_rejects_non_binary_outcomes_and_duplicates():
    assignments, outcomes = _make_experiment(n_per_arm_per_band=5)

    non_binary = outcomes.copy()
    non_binary.loc[0, "outcome"] = 2
    with pytest.raises(ValueError, match="binary"):
        analyze_coupon_experiment(assignments, non_binary)

    duplicated = pd.concat([outcomes, outcomes.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate ids"):
        analyze_coupon_experiment(assignments, duplicated)


def test_main_writes_json_report(tmp_path, capsys):
    assignments, outcomes = _make_experiment(treatment_rate=0.5, control_rate=0.2)
    assignments_path = tmp_path / "assignments.csv"
    outcomes_path = tmp_path / "outcomes.csv"
    report_path = tmp_path / "analysis.json"
    assignments.to_csv(assignments_path, index=False)
    outcomes.to_csv(outcomes_path, index=False)

    main(
        [
            "--assignments", str(assignments_path),
            "--outcomes", str(outcomes_path),
            "--output", str(report_path),
            "--permutations", "200",
            "--random-state", "0",
        ]
    )

    report = json.loads(report_path.read_text())
    assert report["overall"]["difference"] == pytest.approx(0.3)
    assert len(report["by_band"]) == 2
    assert len(report["metadata"]["assignments"]["sha256"]) == 64
    assert report["metadata"]["outcome_column"] == "outcome"

    printed = capsys.readouterr().out
    assert "Intent-to-treat analysis" in printed
    assert "incremental profit" in printed
