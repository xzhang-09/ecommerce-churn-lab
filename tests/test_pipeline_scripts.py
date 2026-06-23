import importlib
import inspect

import pandas as pd
import pytest


def test_prepare_processed_data_uses_distinct_feature_output_path():
    module = importlib.import_module("scripts.prepare_processed_data")

    assert module.CLEANED_OUT.endswith("ecommerce_churn_cleaned.csv")
    assert module.FEATURES_OUT.endswith("ecommerce_churn_features.csv")
    assert module.CLEANED_OUT != module.FEATURES_OUT


def test_run_pipeline_imports_without_great_expectations_installed():
    module = importlib.import_module("scripts.run_pipeline")

    assert callable(module.main)
    assert callable(module.run_data_validation)


def test_run_pipeline_validation_error_explains_how_to_continue_when_dependency_missing():
    module = importlib.import_module("scripts.run_pipeline")

    with pytest.raises(ModuleNotFoundError, match="--skip_validation"):
        module.run_data_validation(pd.DataFrame({"CustomerID": [1]}))


def test_phase2_objective_uses_training_cv_not_holdout_test_data():
    module = importlib.import_module("scripts.test_pipeline_phase2_modeling")

    assert callable(module.make_objective)
    signature = inspect.signature(module.make_objective)
    # Objective is built only from the training split and a CV fold count; it
    # never receives the holdout test data, and tunes a threshold-free metric.
    assert list(signature.parameters) == ["X_train", "y_train", "scoring", "cv_splits", "seed"]


def test_run_pipeline_exposes_reproducible_model_comparison_specs():
    module = importlib.import_module("scripts.run_pipeline")

    specs = module.get_model_specs(scale_pos_weight=3.0, random_state=42)
    names = [spec.name for spec in specs]

    assert names == ["random_forest", "lightgbm", "xgboost"]


def test_run_pipeline_can_select_named_final_model():
    module = importlib.import_module("scripts.run_pipeline")

    model = module.build_model("random_forest", scale_pos_weight=3.0, random_state=42)

    assert model.__class__.__name__ == "RandomForestClassifier"


def test_run_pipeline_builds_lightgbm_model():
    module = importlib.import_module("scripts.run_pipeline")

    model = module.build_model("lightgbm", scale_pos_weight=3.0, random_state=42)

    assert model.__class__.__name__ == "LGBMClassifier"


def test_run_pipeline_metrics_at_threshold_reports_operating_point_and_counts():
    module = importlib.import_module("scripts.run_pipeline")

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
    module = importlib.import_module("scripts.run_pipeline")

    y_true = pd.Series([0, 0, 1, 1])
    probabilities = pd.Series([0.1, 0.4, 0.7, 0.9])

    metrics = module.threshold_free_metrics(y_true, probabilities)

    assert set(metrics) == {"roc_auc", "pr_auc"}
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0


def test_run_pipeline_calibration_report_flags_miscalibration():
    module = importlib.import_module("scripts.run_pipeline")

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
    module = importlib.import_module("scripts.run_pipeline")

    y_true = pd.Series([0, 0, 1, 1])
    probabilities = pd.Series([0.1, 0.4, 0.7, 0.9])

    threshold, value = module.select_threshold(y_true, probabilities, metric="f1")

    # A clean cut exists between the two classes, so perfect F1 is achievable.
    assert value == 1.0
    assert 0.4 < threshold <= 0.7
