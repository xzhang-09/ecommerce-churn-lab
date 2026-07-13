#!/usr/bin/env python3
"""Reproducible churn modeling pipeline.

Runs sequentially: load → validate → preprocess → feature engineering →
train → calibrate → threshold → evaluate, logging everything to MLflow.

This module is the *orchestration* layer. The reusable pieces live in focused
modules and are re-exported here for backward compatibility:

* :mod:`churn.models.estimators` — candidate models (`get_model_specs`, `build_model`)
* :mod:`churn.models.evaluate`   — metrics, threshold selection, calibration
* :mod:`churn.models.training`   — fitting, calibration, cross-validated OOF pass
* :mod:`churn.models.compare`    — the model comparison

``scripts/run_pipeline.py`` is a thin CLI entry point that calls :func:`cli`.
"""

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from churn.data.load_data import load_data
from churn.data.preprocess import impute_numeric, preprocess_data
from churn.features.build_features import apply_feature_schema, build_feature_schema
from churn.logging_utils import configure_logging
from churn.models.tune import build_tuned_estimator, tune_model

# Re-exported so ``churn.pipeline.<name>`` and existing imports keep working
# after the split into focused modules.
from churn.models.estimators import ModelSpec, build_model, get_model_specs
from churn.models.evaluate import (
    calibration_report,
    expected_cost,
    metrics_at_threshold,
    reliability_table_text,
    select_threshold,
    threshold_free_metrics,
)
from churn.models.training import (
    calibrate_prefit,
    cross_val_oof,
    fit_with_early_stopping,
    predict_positive_probability,
)
from churn.models.compare import compare_models
from churn.models.report import (
    default_timestamp,
    git_commit_short,
    render_results_markdown,
)

__all__ = [
    "ModelSpec", "get_model_specs", "build_model",
    "threshold_free_metrics", "metrics_at_threshold", "calibration_report",
    "reliability_table_text", "expected_cost", "select_threshold",
    "predict_positive_probability", "calibrate_prefit", "fit_with_early_stopping",
    "cross_val_oof", "compare_models",
    "PreparedModelSplits", "prepare_model_splits", "run_data_validation",
    "main", "cli", "build_arg_parser",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedModelSplits:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    X_trainval: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    y_trainval: pd.Series
    feature_schema: dict
    feature_cols: list[str]
    medians: pd.Series


def run_data_validation(df, skip_validation: bool = False):
    if skip_validation:
        logger.info("⚠️  Skipping Great Expectations validation (--skip_validation).")
        return True, []

    try:
        from churn.validation.validate_data import validate_ecommerce_data
        return validate_ecommerce_data(df)
    except ModuleNotFoundError as exc:
        if exc.name == "great_expectations":
            raise ModuleNotFoundError(
                "Great Expectations is required for data validation. "
                "Install dependencies with `python -m pip install -e .`, "
                "or rerun with `--skip_validation` for a quick local modeling check."
            ) from exc
        raise


def _coerce_bool_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols):
        df[bool_cols] = df[bool_cols].astype(int)
    return df


def prepare_model_splits(
    df: pd.DataFrame,
    target: str,
    test_size: float,
    val_size: float,
    random_state: int = 42,
) -> PreparedModelSplits:
    """Split first, then fit reusable preprocessing state without test leakage."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in data")

    df_trainval, df_test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target],
        random_state=random_state,
    )
    val_fraction = val_size / (1.0 - test_size)
    df_train, df_val = train_test_split(
        df_trainval,
        test_size=val_fraction,
        stratify=df_trainval[target],
        random_state=random_state,
    )

    feature_schema = build_feature_schema(df_trainval, target_col=target)
    trainval_enc = _coerce_bool_columns(apply_feature_schema(df_trainval, feature_schema))
    train_enc = _coerce_bool_columns(apply_feature_schema(df_train, feature_schema))
    val_enc = _coerce_bool_columns(apply_feature_schema(df_val, feature_schema))
    test_enc = _coerce_bool_columns(apply_feature_schema(df_test, feature_schema))

    feature_cols = list(trainval_enc.drop(columns=[target]).columns)
    X_trainval = trainval_enc.drop(columns=[target])
    y_trainval = trainval_enc[target]
    X_train = train_enc.drop(columns=[target])
    y_train = train_enc[target]
    X_val = val_enc.drop(columns=[target])
    y_val = val_enc[target]
    X_test = test_enc.drop(columns=[target])
    y_test = test_enc[target]

    X_train, medians = impute_numeric(X_train)
    X_val, _ = impute_numeric(X_val, medians)
    X_test, _ = impute_numeric(X_test, medians)
    X_trainval, _ = impute_numeric(X_trainval, medians)

    return PreparedModelSplits(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        X_trainval=X_trainval,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        y_trainval=y_trainval,
        feature_schema=feature_schema,
        feature_cols=feature_cols,
        medians=medians,
    )


# --------------------------------------------------------------------------- #
# Pipeline stages. Each runs inside the active MLflow run started in main().
# --------------------------------------------------------------------------- #

def _configure_mlflow(args) -> str:
    """Point MLflow at local file-based tracking and return the project root.

    Outputs (data/processed, mlruns, artifacts) are written relative to the
    current working directory — the repo root the pipeline is run from — rather
    than relative to this module, which lives deep inside the installed package.
    """
    project_root = os.getcwd()
    mlruns_path = args.mlflow_uri or f"file://{project_root}/mlruns"
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(args.experiment)
    return project_root


def _log_run_config(args) -> None:
    """Log the parameters needed to reproduce the run."""
    mlflow.log_param("model", args.model)
    mlflow.log_param("test_size", args.test_size)
    mlflow.log_param("val_size", args.val_size)
    # The decision threshold is data-driven: when --threshold is omitted it is
    # selected on OOF predictions (logged later as selected_threshold).
    mlflow.log_param("threshold_arg", args.threshold)
    mlflow.log_param("threshold_metric", args.threshold_metric)


def _load_and_validate(args) -> pd.DataFrame:
    """STAGE 1 — load the raw data and run the data-quality gate."""
    logger.info("🔄 Loading data...")
    df = load_data(args.input)
    logger.info(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    logger.info("🔍 Validating data quality with Great Expectations...")
    is_valid, failed = run_data_validation(df, skip_validation=args.skip_validation)
    mlflow.log_metric("data_quality_pass", int(is_valid))
    if not is_valid:
        mlflow.log_text(json.dumps(failed, indent=2), artifact_file="failed_expectations.json")
        raise ValueError(f"❌ Data quality check failed. Issues: {failed}")
    logger.info("✅ Data validation passed. Logged to MLflow.")
    return df


def _preprocess_and_persist(df: pd.DataFrame, project_root: str) -> pd.DataFrame:
    """STAGE 2 — basic cleaning, then persist the cleaned table for debugging."""
    logger.info("🔧 Preprocessing data...")
    df = preprocess_data(df)
    processed_path = os.path.join(project_root, "data", "processed", "ecommerce_churn_cleaned.csv")
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    logger.info(f"✅ Processed dataset saved to {processed_path} | Shape: {df.shape}")
    return df


def _persist_preprocessing_state(
    prepared: PreparedModelSplits, target: str, run_id: str, project_root: str
) -> str:
    """Persist the complete, leakage-free preprocessing state and return the
    artifacts directory.

    The bundle (feature columns + categorical schema + training-fit imputation
    medians) is what lets a later scorer reproduce the exact training transform.
    The medians are fit on the training split only (see prepare_model_splits), so
    saving them here — rather than recomputing per scoring batch — avoids
    train/serve skew. It is stamped with the run id so the model, threshold, and
    preprocessing can be proven to come from the same run.
    """
    artifacts_dir = os.path.join(project_root, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    feature_cols = prepared.feature_cols

    with open(os.path.join(artifacts_dir, "feature_columns.json"), "w") as f:
        json.dump(feature_cols, f)
    feature_schema_path = os.path.join(artifacts_dir, "feature_schema.json")
    with open(feature_schema_path, "w") as f:
        json.dump(prepared.feature_schema, f, indent=2)
    mlflow.log_text("\n".join(feature_cols), artifact_file="feature_columns.txt")
    mlflow.log_artifact(feature_schema_path)

    preprocessing_artifact = {
        "feature_columns": feature_cols,          # Exact feature order
        "feature_schema": prepared.feature_schema,  # Reusable categorical encoding metadata
        "target": target,                          # Target column name
        "mlflow_run_id": run_id,                   # Provenance: ties this bundle to its run
        "medians": prepared.medians.to_dict(),     # Training-fit imputation medians
    }
    joblib.dump(preprocessing_artifact, os.path.join(artifacts_dir, "preprocessing.pkl"))
    mlflow.log_artifact(os.path.join(artifacts_dir, "preprocessing.pkl"))
    mlflow.log_dict(
        {k: float(v) for k, v in prepared.medians.to_dict().items()},
        "imputation_medians.json",
    )
    logger.info(f"✅ Persisted preprocessing state (schema + {len(prepared.medians)} imputation medians)")
    return artifacts_dir


def _build_final_estimator(args, X_train, y_train, scale_pos_weight: float):
    """STAGE 5a — build the final estimator, optionally via an Optuna search."""
    mlflow.log_param("tuned", args.tune)
    mlflow.log_param("calibrate", args.calibrate)
    model = build_model(args.model, scale_pos_weight=scale_pos_weight, random_state=42)

    if not args.tune:
        return model

    if args.model == "random_forest":
        raise ValueError(
            "--tune supports xgboost or lightgbm (random_forest is bagging, "
            "not boosting, so it has no equivalent boosting search)."
        )
    # Optimize a threshold-free ranking metric (default: PR-AUC). The operating
    # threshold is chosen separately on out-of-fold predictions, so tuning and
    # deployment are not scored at different cut points. Avoid a recall-only
    # objective — it flags almost everyone and collapses precision (see tune.py).
    logger.info(f"🔬 Running Optuna search for {args.model} "
                f"(objective={args.tune_metric}, {args.tune_trials} trials)...")
    best_params = tune_model(
        X_train, y_train,
        model_type=args.model,
        scoring=args.tune_metric,
        n_trials=args.tune_trials,
    )
    model = build_tuned_estimator(args.model, best_params, scale_pos_weight, random_state=42)
    mlflow.log_param("tune_metric", args.tune_metric)
    for k, v in best_params.items():
        mlflow.log_param(f"tuned_{k}", v)
    logger.info(f"✅ Using Optuna-tuned hyperparameters: {best_params}")
    return model


def _fit_final_model(model, prepared: PreparedModelSplits) -> float:
    """STAGE 5b — fit the final model (early stopping on the validation split)
    and return the training time."""
    t0 = time.time()
    fit_with_early_stopping(model, prepared.X_train, prepared.y_train,
                            prepared.X_val, prepared.y_val)
    train_time = time.time() - t0
    if hasattr(model, "best_iteration") and model.best_iteration is not None:
        mlflow.log_metric("best_iteration", model.best_iteration)
        logger.info(f"   Early stopping chose {model.best_iteration} trees")
    mlflow.log_metric("train_time", train_time)
    logger.info(f"✅ Model trained in {train_time:.2f} seconds")
    return train_time


def _select_final_threshold(args, y_true, oof_proba) -> float:
    """STAGE 6 — honor an explicit --threshold, else pick it on OOF predictions.

    Never selected on the test set (that would be leakage). Logs the chosen
    threshold to MLflow.
    """
    if args.threshold is not None:
        threshold = args.threshold
        logger.info(f"🎚️  Using fixed decision threshold: {threshold:.3f}")
    elif args.threshold_metric == "cost":
        threshold, neg_cost = select_threshold(
            y_true, oof_proba, metric="cost", fn_fp_ratio=args.fn_fp_ratio
        )
        logger.info(f"🎚️  Selected threshold {threshold:.3f} "
                    f"(CV-OOF cost={-neg_cost:.0f}, FN={args.fn_fp_ratio}xFP)")
    else:
        threshold, oof_score = select_threshold(
            y_true, oof_proba, metric=args.threshold_metric
        )
        logger.info(f"🎚️  Selected threshold {threshold:.3f} "
                    f"(CV-OOF {args.threshold_metric}={oof_score:.3f})")
    mlflow.log_param("selected_threshold", threshold)
    mlflow.log_param("fn_fp_ratio", args.fn_fp_ratio)
    mlflow.log_text(str(threshold), artifact_file="threshold.txt")
    return threshold


def _evaluate_and_log(args, inference_model, prepared: PreparedModelSplits,
                      threshold: float, oof_proba, train_time: float,
                      artifacts_dir: str) -> dict:
    """STAGE 7 — evaluate once on the untouched test split, compute calibration
    and permutation importance, and log everything (incl. the model) to MLflow.

    Returns a dict of the held-out results so the caller can render a committed
    Markdown report (see ``churn.models.report``)."""
    X_test, y_test = prepared.X_test, prepared.y_test
    y_trainval_pos = prepared.y_trainval.reset_index(drop=True)

    logger.info("📊 Evaluating model performance on held-out test set...")
    t1 = time.time()
    proba = predict_positive_probability(inference_model, X_test)  # P(churn)
    pred_time = time.time() - t1
    mlflow.log_metric("pred_time", pred_time)

    ranking = threshold_free_metrics(y_test, proba)
    operating = metrics_at_threshold(y_test, proba, threshold)
    y_pred = (proba >= threshold).astype(int)
    test_cost = expected_cost(operating, args.fn_fp_ratio)

    for metric_name, metric_value in {**ranking, **operating}.items():
        mlflow.log_metric(metric_name, metric_value)
    mlflow.log_metric("test_cost", test_cost)

    logger.info("🎯 Model Performance (test):")
    logger.info(f"   Precision: {operating['precision']:.3f} | "
                f"Recall: {operating['recall']:.3f} | F1: {operating['f1']:.3f}")
    logger.info(f"   ROC AUC: {ranking['roc_auc']:.3f} | PR AUC: {ranking['pr_auc']:.3f}")
    logger.info(f"   Confusion @ {threshold:.3f}: "
                f"TP={operating['tp']} FP={operating['fp']} "
                f"FN={operating['fn']} TN={operating['tn']}")
    logger.info(f"   Business cost (FN={args.fn_fp_ratio}xFP): {test_cost:.0f} "
                f"FP-equivalents [{args.fn_fp_ratio}x{operating['fn']} + {operating['fp']}]")

    # === Probability calibration check (test + OOF) ===
    # Both reflect the post-hoc calibrator applied to the final model (unless
    # --calibrate none). Calibration is monotonic, so it leaves the threshold
    # decisions and PR-AUC untouched while pulling Brier/ECE down.
    brier, ece, table = calibration_report(y_test, proba)
    brier_oof, ece_oof, _ = calibration_report(y_trainval_pos, oof_proba)
    mlflow.log_metric("brier", brier)
    mlflow.log_metric("ece", ece)
    mlflow.log_metric("brier_oof", brier_oof)
    mlflow.log_metric("ece_oof", ece_oof)
    mlflow.log_text(reliability_table_text(table), artifact_file="calibration_reliability_test.csv")
    logger.info("📐 Calibration check (lower = better):")
    logger.info(f"   Brier: {brier:.4f} (test) | {brier_oof:.4f} (OOF)")
    logger.info(f"   ECE:   {ece:.4f} (test) | {ece_oof:.4f} (OOF)")
    if ece_oof > 0.05:
        hint = ("try --calibrate isotonic/sigmoid" if args.calibrate in (None, "none")
                else "consider the other --calibrate method or more calibration data")
        logger.info(f"   ⚠️  ECE > 0.05 — probabilities still not well calibrated ({hint}) "
                    "before using them as true probabilities.")

    # === Feature importance (permutation, on the test split) ===
    # Model-agnostic: how much test PR-AUC drops when each feature is shuffled.
    # More honest than a tree's split-count importance, and it answers the other
    # half of a churn project — not just "who churns" but "which signals drive it".
    logger.info("🔍 Computing permutation importance on the test split...")
    perm = permutation_importance(
        inference_model, X_test, y_test,
        scoring="average_precision", n_repeats=10, random_state=42, n_jobs=-1,
    )
    importance = (
        pd.DataFrame({
            "feature": prepared.feature_cols,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        })
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    importance_path = os.path.join(artifacts_dir, "feature_importance.csv")
    importance.to_csv(importance_path, index=False)
    mlflow.log_artifact(importance_path)
    logger.info("   Top 10 features by permutation importance (PR-AUC drop):")
    logger.info(importance.head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # === Model serialization ===
    # Log the calibrated estimator actually used for inference (falls back to the
    # bare model when --calibrate none) so the artifact reproduces the evaluated
    # probabilities exactly.
    logger.info("💾 Saving model to MLflow...")
    mlflow.sklearn.log_model(inference_model, artifact_path="model")
    logger.info("✅ Model saved to MLflow")

    logger.info("\n⏱️  Performance Summary:")
    logger.info(f"   Training time: {train_time:.2f}s")
    logger.info(f"   Inference time: {pred_time:.4f}s")
    logger.info(f"   Samples per second: {len(X_test) / pred_time:.0f}")
    logger.info("\n📈 Detailed Classification Report:")
    logger.info("\n" + classification_report(y_test, y_pred, digits=3))

    return {
        "ranking": ranking,
        "operating": operating,
        "threshold": threshold,
        "fn_fp_ratio": args.fn_fp_ratio,
        "test_cost": test_cost,
        "brier": brier, "ece": ece,
        "brier_oof": brier_oof, "ece_oof": ece_oof,
        "importance": importance,
        "test_rows": int(len(y_test)),
        "test_pos": int((y_test == 1).sum()),
        "test_neg": int((y_test == 0).sum()),
    }


def _write_results_report(results: dict, project_root: str) -> None:
    """STAGE 8 — render the held-out results to ``docs/RESULTS.md`` (committed,
    so the README can link to it) and also log it as an MLflow artifact."""
    markdown = render_results_markdown(results)
    docs_dir = os.path.join(project_root, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    report_path = os.path.join(docs_dir, "RESULTS.md")
    with open(report_path, "w") as f:
        f.write(markdown)
    mlflow.log_text(markdown, artifact_file="RESULTS.md")
    logger.info(f"📝 Wrote results report to {report_path}")


def main(args):
    """Orchestrate the complete ML workflow as a sequence of stages, all logged
    under a single MLflow run."""
    configure_logging()
    project_root = _configure_mlflow(args)

    with mlflow.start_run() as active_run:
        run_id = active_run.info.run_id
        _log_run_config(args)

        # STAGE 1-2: load + validate + preprocess
        df = _load_and_validate(args)
        raw_rows = int(len(df))
        df = _preprocess_and_persist(df, project_root)
        clean_rows = int(len(df))

        # STAGE 3-4: split, fit reusable preprocessing state, persist it
        logger.info("🛠️  Building features and splitting (train / validation / test)...")
        target = args.target
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in data")
        prepared = prepare_model_splits(
            df, target=target, test_size=args.test_size, val_size=args.val_size,
            random_state=42,
        )
        logger.info(f"✅ Feature engineering completed: {len(prepared.feature_cols) + 1} features")
        logger.info(f"✅ Train: {prepared.X_train.shape[0]} | Validation: {prepared.X_val.shape[0]} "
                    f"| Test: {prepared.X_test.shape[0]} samples")
        artifacts_dir = _persist_preprocessing_state(prepared, target, run_id, project_root)

        # Class imbalance: scale_pos_weight rebalances XGBoost's loss (RF/LGBM use
        # class_weight). It de-calibrates probabilities, so the operating threshold
        # is selected from data (below), not hard-coded on top of the reweighting.
        scale_pos_weight = (prepared.y_train == 0).sum() / (prepared.y_train == 1).sum()
        logger.info(f"📈 Class imbalance ratio: {scale_pos_weight:.2f} (applied to positive class)")

        comparison = None
        if args.compare_models:
            comparison = compare_models(
                prepared.X_trainval, prepared.y_trainval,
                scale_pos_weight=scale_pos_weight,
                threshold_metric=args.threshold_metric,
                fn_fp_ratio=args.fn_fp_ratio,
            )
            comparison_path = os.path.join(artifacts_dir, "model_comparison.csv")
            comparison.to_csv(comparison_path, index=False)
            mlflow.log_artifact(comparison_path)

        # STAGE 5: build + cross-validate + fit the final model
        logger.info(f"🤖 Training final model: {args.model}")
        model = _build_final_estimator(args, prepared.X_train, prepared.y_train, scale_pos_weight)

        # One 5-fold CV pass over train+val gives both stability bands AND every
        # row's OOF probability, which is what the decision threshold is chosen on.
        logger.info("🔁 Running 5-fold CV (out-of-fold predictions, early stopping per fold)...")
        oof_proba, cv_stats = cross_val_oof(
            model, prepared.X_trainval, prepared.y_trainval, cv=5, calibrate=args.calibrate
        )
        for cv_metric, (mean, std) in cv_stats.items():
            mlflow.log_metric(f"cv_{cv_metric}_mean", mean)
            mlflow.log_metric(f"cv_{cv_metric}_std", std)
            logger.info(f"   CV {cv_metric}: {mean:.3f} ± {std:.3f}")

        train_time = _fit_final_model(model, prepared)

        # Post-hoc calibration on the validation split — this is the estimator
        # used for every prediction below.
        inference_model = calibrate_prefit(model, prepared.X_val, prepared.y_val, args.calibrate)
        if args.calibrate not in (None, "none"):
            logger.info(f"📏 Calibrated final probabilities with {args.calibrate} "
                        f"(fit on the validation split)")

        # STAGE 6-7: threshold selection + final evaluation
        y_trainval_pos = prepared.y_trainval.reset_index(drop=True)
        threshold = _select_final_threshold(args, y_trainval_pos, oof_proba)
        results = _evaluate_and_log(args, inference_model, prepared, threshold,
                                    oof_proba, train_time, artifacts_dir)

        # STAGE 8: render the committed Markdown results report the README links to.
        results.update({
            "model": args.model,
            "run_id": run_id,
            "git_commit": git_commit_short(project_root),
            "timestamp": default_timestamp(),
            "dataset": {
                "raw_rows": raw_rows,
                "clean_rows": clean_rows,
                "churn_rate": float((df[target] == 1).mean()),
                "target": target,
            },
            "cv_stats": cv_stats,
            "comparison": comparison,
        })
        _write_results_report(results, project_root)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the pipeline's CLI argument parser (factored out so the thin
    ``scripts/run_pipeline.py`` wrapper and the ``churn-pipeline`` console entry
    point share one source of truth)."""
    p = argparse.ArgumentParser(description="Run the e-commerce churn modeling pipeline")
    p.add_argument("--input", type=str, required=True,
                   help="path to raw data (e.g., data/raw/E Commerce Dataset.xlsx)")
    p.add_argument("--target", type=str, default="Churn")
    p.add_argument("--threshold", type=float, default=None,
                   help="fixed decision threshold; if omitted, the threshold is "
                        "selected on the validation split to maximize --threshold_metric")
    p.add_argument("--threshold_metric", type=str, default="cost",
                   choices=["cost", "f1", "precision", "recall"],
                   help="objective for auto-selecting the decision threshold on validation; "
                        "'cost' minimizes the FN/FP business cost (default), the others maximize that metric")
    p.add_argument("--fn_fp_ratio", type=float, default=10.0,
                   help="business cost of a missed churner (FN) relative to a false alarm (FP); "
                        "used by the 'cost' threshold objective and the reported business cost (default: 10)")
    p.add_argument("--model", type=str, default="lightgbm",
                   choices=["random_forest", "lightgbm", "xgboost"],
                   help="final model to train after optional model comparison "
                        "(default: lightgbm — lowest CV PR-AUC rank tie-breaker and lowest FN/FP business cost)")
    p.add_argument("--test_size", type=float, default=0.2,
                   help="fraction of the full dataset held out as the test set")
    p.add_argument("--val_size", type=float, default=0.2,
                   help="fraction of the full dataset used as the validation set (threshold/model selection)")
    p.add_argument("--experiment", type=str, default="E-Commerce Churn")
    p.add_argument("--mlflow_uri", type=str, default=None,
                    help="override MLflow tracking URI, else uses project_root/mlruns")
    p.add_argument("--tune", action="store_true",
                   help="run an Optuna hyperparameter search for the final model "
                        "(xgboost or lightgbm) instead of using the fixed defaults")
    p.add_argument("--tune_metric", type=str, default="average_precision",
                   help="sklearn scoring metric Optuna optimizes (default: average_precision / "
                        "PR-AUC, a threshold-free metric — see tune.py)")
    p.add_argument("--tune_trials", type=int, default=20)
    p.add_argument("--calibrate", type=str, default="isotonic",
                   choices=["isotonic", "sigmoid", "none"],
                   help="post-hoc probability calibration for the final model, fit on the "
                        "validation split ('isotonic' default; 'sigmoid'/Platt is steadier on "
                        "very small data; 'none' keeps the raw, de-calibrated scores). "
                        "Monotonic, so it does not change ranking or the selected threshold")
    p.add_argument("--skip_validation", action="store_true",
                   help="skip Great Expectations validation when the optional validation dependency is unavailable")
    p.add_argument("--compare_models", action="store_true",
                   help="run a fair, cross-validated comparison of all candidate models "
                        "(RandomForest/LightGBM/XGBoost) before training the selected final model")
    return p


def cli(argv=None) -> None:
    """Console entry point: parse args and run the pipeline.

    Examples
    --------
    Default run::

        python scripts/run_pipeline.py --input "data/raw/E Commerce Dataset.xlsx" --target Churn

    With an Optuna search instead of the fixed defaults::

        python scripts/run_pipeline.py --input "data/raw/E Commerce Dataset.xlsx" \\
            --target Churn --tune --tune_metric average_precision --tune_trials 20
    """
    main(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    cli()
