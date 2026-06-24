import importlib.util

import pandas as pd
import pytest

from churn import pipeline
from churn.data import prepare


def test_prepare_processed_data_uses_distinct_feature_output_path():
    assert prepare.CLEANED_OUT.endswith("ecommerce_churn_cleaned.csv")
    assert prepare.FEATURES_OUT.endswith("ecommerce_churn_features.csv")
    assert prepare.CLEANED_OUT != prepare.FEATURES_OUT


def test_pipeline_exposes_callable_entry_points():
    assert callable(pipeline.main)
    assert callable(pipeline.cli)
    assert callable(pipeline.run_data_validation)
    assert callable(pipeline.build_arg_parser)


@pytest.mark.skipif(
    importlib.util.find_spec("great_expectations") is not None,
    reason="great_expectations is installed, so the missing-dependency path isn't exercised",
)
def test_run_pipeline_validation_error_explains_how_to_continue_when_dependency_missing():
    with pytest.raises(ModuleNotFoundError, match="--skip_validation"):
        pipeline.run_data_validation(pd.DataFrame({"CustomerID": [1]}))


def test_run_pipeline_exposes_reproducible_model_comparison_specs():
    specs = pipeline.get_model_specs(scale_pos_weight=3.0, random_state=42)
    names = [spec.name for spec in specs]

    assert names == ["random_forest", "lightgbm", "xgboost"]


def test_run_pipeline_can_select_named_final_model():
    module = pipeline

    model = module.build_model("random_forest", scale_pos_weight=3.0, random_state=42)

    assert model.__class__.__name__ == "RandomForestClassifier"


def test_run_pipeline_builds_lightgbm_model():
    module = pipeline

    model = module.build_model("lightgbm", scale_pos_weight=3.0, random_state=42)

    assert model.__class__.__name__ == "LGBMClassifier"


def test_run_pipeline_metrics_at_threshold_reports_operating_point_and_counts():
    module = pipeline

    y_true = pd.Series([0, 0, 1, 1])
    probabilities = pd.Series([0.1, 0.4, 0.7, 0.9])

    metrics = module.metrics_at_threshold(y_true, probabilities, threshold=0.5)

    assert set(metrics) == {"precision", "recall", "f1", "tn", "fp", "fn", "tp"}
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    # Perfect separation at 0.5: both positives caught, no false alarms.
    assert (metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]) == (2, 0, 0, 2)


def test_run_pipeline_threshold_free_metrics_include_pr_auc():
    module = pipeline

    y_true = pd.Series([0, 0, 1, 1])
    probabilities = pd.Series([0.1, 0.4, 0.7, 0.9])

    metrics = module.threshold_free_metrics(y_true, probabilities)

    assert set(metrics) == {"roc_auc", "pr_auc"}
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0


def test_run_pipeline_calibration_report_flags_miscalibration():
    module = pipeline

    # Perfectly calibrated: predict 0.0 for negatives and 1.0 for positives.
    y_true = pd.Series([0, 0, 1, 1])
    perfect = pd.Series([0.0, 0.0, 1.0, 1.0])
    brier, ece, table = module.calibration_report(y_true, perfect)
    assert brier == 0.0
    assert ece == 0.0
    assert all(len(row) == 5 for row in table)

    # Badly miscalibrated: confident but wrong -> high Brier and ECE.
    wrong = pd.Series([0.9, 0.9, 0.1, 0.1])
    brier_bad, ece_bad, _ = module.calibration_report(y_true, wrong)
    assert brier_bad > brier
    assert ece_bad > 0.5


def test_run_pipeline_select_threshold_picks_value_maximizing_metric():
    module = pipeline

    y_true = pd.Series([0, 0, 1, 1])
    probabilities = pd.Series([0.1, 0.4, 0.7, 0.9])

    threshold, value = module.select_threshold(y_true, probabilities, metric="f1")

    # A clean cut exists between the two classes, so perfect F1 is achievable.
    assert value == 1.0
    assert 0.4 < threshold <= 0.7
