#!/usr/bin/env python3
"""Reproducible churn modeling pipeline.

Runs sequentially: load → validate → preprocess → feature engineering →
train → calibrate → threshold → evaluate, logging everything to MLflow.

This is the importable library form of the pipeline; ``scripts/run_pipeline.py``
is a thin CLI entry point that calls :func:`cli` here.
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from dataclasses import dataclass
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix,
    brier_score_loss
)
from xgboost import XGBClassifier

# Local package modules — core pipeline components.
from churn.data.load_data import load_data                    # Data loading with error handling
from churn.data.preprocess import preprocess_data            # Basic data cleaning
from churn.data.preprocess import impute_numeric            # Leakage-free median imputation
from churn.features.build_features import (
    apply_feature_schema,
    build_feature_schema,
)  # Feature engineering (CRITICAL for model performance)
from churn.models.tune import tune_model, build_tuned_estimator  # Optional Optuna hyperparameter search


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: object


def get_model_specs(scale_pos_weight: float, random_state: int = 42):
    """Candidate estimators with plain, conventional defaults.

    The previous XGBoost config used oddly specific magic numbers
    (n_estimators=301, learning_rate=0.034, ...) left over from an old tuning
    run with no provenance. These are replaced with standard starting points:
    a low learning rate (0.05), moderate depth, and 0.8 row/column subsampling.
    For the boosting models `n_estimators` is intentionally a high *upper bound*
    — early stopping (see fit_with_early_stopping) picks the real tree count on a
    validation fold, so the number of trees is data-driven rather than hardcoded.
    RandomForest has no early stopping (bagging, not boosting), so its tree count
    is fixed.
    """
    return [
        ModelSpec(
            "random_forest",
            RandomForestClassifier(
                n_estimators=400,
                class_weight="balanced",
                n_jobs=-1,
                random_state=random_state,
            ),
        ),
        ModelSpec(
            "lightgbm",
            LGBMClassifier(
                n_estimators=2000,          # upper bound; early stopping decides
                learning_rate=0.05,
                class_weight="balanced",
                n_jobs=-1,
                random_state=random_state,
                verbosity=-1,
            ),
        ),
        ModelSpec(
            "xgboost",
            XGBClassifier(
                n_estimators=2000,          # upper bound; early stopping decides
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                random_state=random_state,
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
            ),
        ),
    ]


def build_model(model_name: str, scale_pos_weight: float, random_state: int = 42):
    specs = {
        spec.name: spec.estimator
        for spec in get_model_specs(scale_pos_weight=scale_pos_weight, random_state=random_state)
    }
    if model_name not in specs:
        valid = ", ".join(specs)
        raise ValueError(f"Unknown model '{model_name}'. Choose one of: {valid}")
    return specs[model_name]


def threshold_free_metrics(y_true, probabilities):
    """Ranking-quality metrics that do not depend on the decision threshold.

    For imbalanced churn (~17% positives) PR-AUC (average precision) is more
    informative than ROC AUC, which can look strong while precision is poor — we
    report both.
    """
    return {
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
    }


def metrics_at_threshold(y_true, probabilities, threshold: float):
    """Operating-point metrics at a specific decision threshold, including the
    raw confusion-matrix counts so the business cost of FN vs FP is recoverable."""
    y_pred = (np.asarray(probabilities) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def calibration_report(y_true, probabilities, n_bins: int = 10):
    """Assess how well predicted probabilities match observed frequencies.

    Class reweighting (scale_pos_weight / class_weight) deliberately distorts the
    probability scale to shift the decision boundary, so the raw scores are NOT
    reliable probabilities. This quantifies that distortion with:
      * Brier score — mean squared error of the probabilities (lower is better);
      * ECE (Expected Calibration Error) — average gap between predicted
        confidence and observed accuracy across `n_bins` probability bins,
        weighted by bin population (0 = perfectly calibrated).
    Returns ``(brier, ece, reliability_table)`` where the table has one row per
    non-empty bin: (bin_low, bin_high, count, mean_predicted, observed_rate).
    """
    y_true = np.asarray(y_true, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    brier = brier_score_loss(y_true, probabilities)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(probabilities, edges[1:-1])
    ece = 0.0
    table = []
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        mean_pred = float(probabilities[mask].mean())
        observed = float(y_true[mask].mean())
        ece += (count / len(y_true)) * abs(observed - mean_pred)
        table.append((float(edges[b]), float(edges[b + 1]), count, mean_pred, observed))
    return brier, ece, table


def reliability_table_text(table):
    """Render the reliability table from calibration_report as CSV text."""
    lines = ["bin_low,bin_high,count,mean_predicted,observed_rate"]
    for low, high, count, mean_pred, observed in table:
        lines.append(f"{low:.2f},{high:.2f},{count},{mean_pred:.4f},{observed:.4f}")
    return "\n".join(lines)


def expected_cost(operating, fn_fp_ratio: float):
    """Business cost at an operating point, in units of one false positive.

    A missed churner (FN) is treated as `fn_fp_ratio` times as expensive as a
    false alarm (FP) — e.g. fn_fp_ratio=10 means letting a churner slip away
    costs as much as wrongly targeting 10 loyal customers. True positives /
    negatives carry no cost in this simple model.
    """
    return fn_fp_ratio * operating["fn"] + operating["fp"]


def _threshold_objective(operating, metric: str, fn_fp_ratio: float):
    """Score (higher = better) used to pick a threshold. For the cost objective
    we maximize the negative business cost; otherwise the named metric itself."""
    if metric == "cost":
        return -expected_cost(operating, fn_fp_ratio)
    return operating[metric]


def select_threshold(y_true, probabilities, metric: str = "cost", fn_fp_ratio: float = 10.0):
    """Pick the decision threshold that optimizes `metric` on the given
    (validation) data. `metric="cost"` minimizes the FN/FP business cost
    (default, FN is `fn_fp_ratio`x an FP); "f1"/"precision"/"recall" maximize
    that metric instead. The threshold must be chosen on held-out data that is
    not the test set — picking it on the test set is leakage and inflates the
    reported score. Returns ``(threshold, best_objective_value)`` where the
    objective is the quantity being maximized (negative cost for "cost").
    """
    probabilities = np.asarray(probabilities)
    # Candidate thresholds = the distinct predicted probabilities; every
    # achievable confusion matrix is realized at one of these cut points.
    candidates = np.unique(np.round(probabilities, 4))
    best_t, best_v = 0.5, -np.inf
    for t in candidates:
        operating = metrics_at_threshold(y_true, probabilities, t)
        value = _threshold_objective(operating, metric, fn_fp_ratio)
        if value > best_v:
            best_v, best_t = value, float(t)
    return best_t, best_v


def predict_positive_probability(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1 / (1 + np.exp(-scores))
    return model.predict(X)


def calibrate_prefit(estimator, X_cal, y_cal, method: str | None):
    """Wrap an already-fitted `estimator` in a probability calibrator.

    Class reweighting (scale_pos_weight / class_weight) distorts the probability
    scale to move the decision boundary, so the raw scores are good for *ranking*
    but are not trustworthy probabilities. A post-hoc calibrator fitted on a
    held-out slice (`cv="prefit"`) restores that scale without retraining the
    base model. Calibration is monotonic, so it preserves ranking (PR-AUC/ROC-AUC)
    while making the probabilities usable for expected-value targeting.

    `method` is "isotonic", "sigmoid" (Platt), or None/"none" to skip calibration
    and return the estimator unchanged. The calibration slice does double duty as
    the early-stopping set; reusing it adds a little optimism to the fitted
    calibration map but avoids carving out yet another split from a small dataset.
    """
    if method in (None, "none"):
        return estimator
    calibrated = CalibratedClassifierCV(estimator, method=method, cv="prefit")
    calibrated.fit(X_cal, y_cal)
    return calibrated


def fit_with_early_stopping(estimator, X_tr, y_tr, X_es, y_es, rounds: int = 50):
    """Fit `estimator`, using early stopping for the boosting models so the tree
    count is chosen from data instead of a hardcoded `n_estimators`.

    XGBoost and LightGBM stop adding trees once the held-out (`X_es`) metric stops
    improving for `rounds` iterations. RandomForest is bagging, not boosting, so
    it has no early-stopping notion and is fit as-is.
    """
    name = estimator.__class__.__name__
    if name == "XGBClassifier":
        estimator.set_params(early_stopping_rounds=rounds)
        estimator.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)
    elif name == "LGBMClassifier":
        from lightgbm import early_stopping as lgb_early_stopping
        estimator.fit(
            X_tr, y_tr,
            eval_set=[(X_es, y_es)],
            callbacks=[lgb_early_stopping(rounds, verbose=False)],
        )
    else:
        estimator.fit(X_tr, y_tr)
    return estimator


# Threshold-free scorers used for cross-validated model ranking.
_RANK_SCORERS = {
    "pr_auc": average_precision_score,
    "roc_auc": roc_auc_score,
}


def cross_val_oof(estimator, X, y, cv: int = 5, es_rounds: int = 50,
                  random_state: int = 42, calibrate: str | None = None):
    """One stratified CV pass returning both out-of-fold (OOF) probabilities and
    per-fold threshold-free ranking scores.

    Each fold is predicted by a model that never saw it; within every fold the
    training portion is further split into a sub-train and a small early-stopping
    set, so early stopping never peeks at the fold being predicted — keeping both
    the OOF probabilities and the CV scores honest. Returns
    ``(oof_proba, {metric: (mean, std)})``.

    The OOF probabilities let the decision threshold be chosen on every row's
    held-out prediction rather than on a single validation split, which makes the
    selected threshold far less sensitive to one lucky/unlucky split.

    The inner early-stopping split is seeded per fold (``random_state + fold``) so
    the tree count isn't picked on the same sub-split in every fold — this
    decorrelates early stopping from the final model's own validation split, which
    shares the base ``random_state``.

    `calibrate` ("isotonic"/"sigmoid"/None) optionally fits a probability
    calibrator on each fold's early-stopping slice before predicting that fold,
    so the OOF probabilities (and any threshold or calibration diagnostics derived
    from them) are on the same calibrated scale as the final model. Ranking-based
    CV scores are unaffected because calibration is monotonic.
    """
    y = y.reset_index(drop=True)
    X = X.reset_index(drop=True)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    oof = np.zeros(len(y))
    fold_scores = {m: [] for m in _RANK_SCORERS}
    for fold, (fit_idx, pred_idx) in enumerate(skf.split(X, y)):
        X_fold, y_fold = X.iloc[fit_idx], y.iloc[fit_idx]
        X_pred, y_pred = X.iloc[pred_idx], y.iloc[pred_idx]
        X_sub, X_es, y_sub, y_es = train_test_split(
            X_fold, y_fold, test_size=0.2, stratify=y_fold,
            random_state=random_state + fold,
        )
        est = fit_with_early_stopping(clone(estimator), X_sub, y_sub, X_es, y_es, es_rounds)
        est = calibrate_prefit(est, X_es, y_es, calibrate)
        proba = predict_positive_probability(est, X_pred)
        oof[pred_idx] = proba
        for metric, scorer in _RANK_SCORERS.items():
            fold_scores[metric].append(scorer(y_pred, proba))
    stats = {m: (float(np.mean(v)), float(np.std(v))) for m, v in fold_scores.items()}
    return oof, stats


def compare_models(
    X_trainval,
    y_trainval,
    scale_pos_weight: float,
    threshold_metric: str = "cost",
    fn_fp_ratio: float = 10.0,
    cv: int = 5,
    random_state: int = 42,
):
    """Fair, cross-validated comparison of the candidate models.

    Fairness fixes over the previous version:
      * All candidates are compared on equal footing at their default configs —
        no per-model Optuna for one model only (Optuna is applied to the final
        model separately, never inside the ranking). Tuning only XGBoost here
        would mean "tuned XGB vs untuned rivals".
      * Models are ranked by cross-validated PR-AUC (mean over `cv` folds), not a
        single noisy train/test split. The std is reported alongside so close
        ranks can be read as ties.
      * Boosting models use early stopping inside CV (see cross_val_oof).

    The precision/recall/f1/cost columns are reported per model for context at
    each model's out-of-fold threshold, computed on the same out-of-fold
    predictions — never on the held-out test set, which is reserved for the single
    final model. They do NOT drive the ranking. (The comparison stays calibration-
    agnostic: ranking and the cost-optimal threshold are invariant to the monotonic
    calibration applied to the final model.)
    """
    y_trainval_pos = y_trainval.reset_index(drop=True)
    rows = []
    for spec in get_model_specs(scale_pos_weight=scale_pos_weight, random_state=random_state):
        print(f"🔎 Cross-validating candidate model: {spec.name}")
        t0 = time.time()
        oof, cv_scores = cross_val_oof(
            spec.estimator, X_trainval, y_trainval, cv=cv, random_state=random_state
        )
        cv_time = time.time() - t0

        # Threshold and operating point both come from the out-of-fold predictions
        # (every row scored by a model that never saw it), so the context metrics
        # need no extra single-split refit and never touch the test set.
        threshold, _ = select_threshold(
            y_trainval_pos, oof, metric=threshold_metric, fn_fp_ratio=fn_fp_ratio
        )
        operating = metrics_at_threshold(y_trainval_pos, oof, threshold)

        pr_mean, pr_std = cv_scores["pr_auc"]
        roc_mean, roc_std = cv_scores["roc_auc"]
        rows.append({
            "model": spec.name,
            "cv_pr_auc": pr_mean, "cv_pr_std": pr_std,
            "cv_roc_auc": roc_mean, "cv_roc_std": roc_std,
            "threshold": threshold,
            "precision": operating["precision"],
            "recall": operating["recall"],
            "f1": operating["f1"],
            "oof_cost": expected_cost(operating, fn_fp_ratio),
            "cv_time": cv_time,
        })

        with mlflow.start_run(run_name=f"compare_{spec.name}", nested=True):
            mlflow.log_param("model", spec.name)
            mlflow.log_param("threshold", threshold)
            mlflow.log_param("threshold_metric", threshold_metric)
            mlflow.log_metric("cv_pr_auc_mean", pr_mean)
            mlflow.log_metric("cv_pr_auc_std", pr_std)
            mlflow.log_metric("cv_roc_auc_mean", roc_mean)
            mlflow.log_metric("cv_roc_auc_std", roc_std)
            for metric_name in ("precision", "recall", "f1"):
                mlflow.log_metric(metric_name, operating[metric_name])
            mlflow.log_metric("oof_cost", expected_cost(operating, fn_fp_ratio))

    # Rank on cross-validated PR-AUC (threshold-free, prevalence-aware).
    results = pd.DataFrame(rows).sort_values(["cv_pr_auc", "cv_roc_auc"], ascending=False)
    print(f"\n📊 Model comparison — ranked by CV PR-AUC (mean ± std over folds); "
          f"precision/recall/f1 and oof_cost (FN={fn_fp_ratio}xFP) are out-of-fold "
          f"context at each model's threshold:")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    best = results.iloc[0]
    print(f"\n🏆 Best by CV PR-AUC: {best['model']} "
          f"({best['cv_pr_auc']:.3f} ± {best['cv_pr_std']:.3f})")
    return results


def run_data_validation(df, skip_validation: bool = False):
    if skip_validation:
        print("⚠️  Skipping Great Expectations validation (--skip_validation).")
        return True, []

    try:
        from churn.validation.validate_data import validate_ecommerce_data
    except ModuleNotFoundError as exc:
        if exc.name == "great_expectations":
            raise ModuleNotFoundError(
                "Great Expectations is required for data validation. "
                "Install dependencies with `python -m pip install -r requirements.txt`, "
                "or rerun with `--skip_validation` for a quick local modeling check."
            ) from exc
        raise

    return validate_ecommerce_data(df)

def main(args):
    """
    Main training pipeline function that orchestrates the complete ML workflow.
    
    """
    
    # === MLflow Setup - ESSENTIAL for experiment tracking ===
    # Configure MLflow to use local file-based tracking (not a tracking server).
    # Outputs (data/processed, mlruns, artifacts) are written relative to the
    # current working directory — i.e. the repo root the pipeline is run from —
    # rather than relative to this module's location, which now lives several
    # levels deep inside the installed `churn` package.
    project_root = os.getcwd()
    mlruns_path = args.mlflow_uri or f"file://{project_root}/mlruns"  # Local file-based tracking
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(args.experiment)  # Creates experiment if doesn't exist

    # Start MLflow run - all subsequent logging will be tracked under this run
    with mlflow.start_run():
        # === Log hyperparameters and configuration ===
        # REQUIRED: These parameters are essential for model reproducibility
        mlflow.log_param("model", args.model)          # Final model type
        mlflow.log_param("test_size", args.test_size)   # Test split fraction
        mlflow.log_param("val_size", args.val_size)     # Validation split fraction
        # The decision threshold is data-driven: when --threshold is omitted it is
        # selected on the validation split (logged later as selected_threshold).
        mlflow.log_param("threshold_arg", args.threshold)
        mlflow.log_param("threshold_metric", args.threshold_metric)

        # === STAGE 1: Data Loading & Validation ===
        print("🔄 Loading data...")
        df = load_data(args.input)  # Load raw CSV data with error handling
        print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        # === CRITICAL: Data Quality Validation ===
        # This step is ESSENTIAL for production ML - validates data quality before training
        print("🔍 Validating data quality with Great Expectations...")
        is_valid, failed = run_data_validation(df, skip_validation=args.skip_validation)
        mlflow.log_metric("data_quality_pass", int(is_valid))  # Track data quality over time

        if not is_valid:
            # Log validation failures for debugging
            import json
            mlflow.log_text(json.dumps(failed, indent=2), artifact_file="failed_expectations.json")
            raise ValueError(f"❌ Data quality check failed. Issues: {failed}")
        else:
            print("✅ Data validation passed. Logged to MLflow.")

        # === STAGE 2: Data Preprocessing ===
        print("🔧 Preprocessing data...")
        df = preprocess_data(df)  # Basic cleaning (handle missing values, fix data types)

        # Save processed dataset for reproducibility and debugging
        processed_path = os.path.join(project_root, "data", "processed", "ecommerce_churn_cleaned.csv")
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_csv(processed_path, index=False)
        print(f"✅ Processed dataset saved to {processed_path} | Shape: {df.shape}")

        # === STAGE 3: Feature Engineering - CRITICAL for Model Performance ===
        print("🛠️  Building features...")
        target = args.target
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in data")
        
        # Fit feature metadata on this training dataset and apply it immediately.
        # The schema is saved below so future batches can be encoded with the
        # same columns and category mappings.
        feature_schema = build_feature_schema(df, target_col=target)
        df_enc = apply_feature_schema(df, feature_schema)
        
        # IMPORTANT: Convert boolean columns to integers for XGBoost compatibility
        for c in df_enc.select_dtypes(include=["bool"]).columns:
            df_enc[c] = df_enc[c].astype(int)
        print(f"✅ Feature engineering completed: {df_enc.shape[1]} features")

        # === Save Feature Metadata for Reproducibility ===
        # This records the exact feature columns used in the model run.
        import json, joblib
        artifacts_dir = os.path.join(project_root, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        # Get feature columns (exclude target)
        feature_cols = list(df_enc.drop(columns=[target]).columns)
        
        # Save locally for analysis and reproducibility.
        with open(os.path.join(artifacts_dir, "feature_columns.json"), "w") as f:
            json.dump(feature_cols, f)
        feature_schema_path = os.path.join(artifacts_dir, "feature_schema.json")
        with open(feature_schema_path, "w") as f:
            json.dump(feature_schema, f, indent=2)

        # Log to MLflow with the model run.
        mlflow.log_text("\n".join(feature_cols), artifact_file="feature_columns.txt")
        mlflow.log_artifact(feature_schema_path)

        # Save preprocessing metadata for experiment reproducibility.
        preprocessing_artifact = {
            "feature_columns": feature_cols,  # Exact feature order
            "feature_schema": feature_schema,  # Reusable categorical encoding metadata
            "target": target                  # Target column name
        }
        joblib.dump(preprocessing_artifact, os.path.join(artifacts_dir, "preprocessing.pkl"))
        mlflow.log_artifact(os.path.join(artifacts_dir, "preprocessing.pkl"))
        print(f"✅ Saved {len(feature_cols)} feature columns for reproducibility")

        # === STAGE 4: Train / Validation / Test Split ===
        # Three-way split: train fits the model, validation selects the decision
        # threshold (and any model choice), test is touched only once for the
        # final reported metrics. A single train/test split would force the
        # threshold to be tuned on the same data it is scored on (leakage).
        print("📊 Splitting data (train / validation / test)...")
        X = df_enc.drop(columns=[target])  # Feature matrix
        y = df_enc[target]                 # Target vector

        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y,
            test_size=args.test_size,
            stratify=y,
            random_state=42,
        )
        # Carve the validation slice out of the remaining train+val pool so its
        # size stays a fixed fraction of the *full* dataset.
        val_fraction = args.val_size / (1.0 - args.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=val_fraction,
            stratify=y_trainval,
            random_state=42,
        )

        # === Leakage-free median imputation ===
        # Medians are fit on the training split only, then applied to validation
        # and test (and the trainval pool used for CV stability below). The
        # *_missing indicator columns added in preprocessing remain intact.
        X_train, medians = impute_numeric(X_train)
        X_val, _ = impute_numeric(X_val, medians)
        X_test, _ = impute_numeric(X_test, medians)
        X_trainval, _ = impute_numeric(X_trainval, medians)
        print(
            f"✅ Train: {X_train.shape[0]} | Validation: {X_val.shape[0]} "
            f"| Test: {X_test.shape[0]} samples"
        )

        # === Handle class imbalance ===
        # scale_pos_weight rebalances XGBoost's loss (RF/LGBM use class_weight).
        # NOTE: this improves ranking/recall but de-calibrates probabilities, so
        # we do NOT also hard-code a low threshold on top of it — the operating
        # threshold is selected from data on the validation split below.
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"📈 Class imbalance ratio: {scale_pos_weight:.2f} (applied to positive class)")

        if args.compare_models:
            comparison = compare_models(
                X_trainval,
                y_trainval,
                scale_pos_weight=scale_pos_weight,
                threshold_metric=args.threshold_metric,
                fn_fp_ratio=args.fn_fp_ratio,
            )
            comparison_path = os.path.join(artifacts_dir, "model_comparison.csv")
            comparison.to_csv(comparison_path, index=False)
            mlflow.log_artifact(comparison_path)

        # === STAGE 5: Model Training ===
        print(f"🤖 Training final model: {args.model}")

        model = build_model(args.model, scale_pos_weight=scale_pos_weight, random_state=42)
        mlflow.log_param("tuned", args.tune)
        mlflow.log_param("calibrate", args.calibrate)

        if args.tune:
            if args.model == "random_forest":
                raise ValueError(
                    "--tune supports xgboost or lightgbm (random_forest is bagging, "
                    "not boosting, so it has no equivalent boosting search)."
                )
            # Optimize a threshold-free ranking metric (default: PR-AUC). The
            # operating threshold is chosen separately on out-of-fold predictions,
            # so tuning and deployment are no longer scored at different cut
            # points. Avoid a recall-only objective — it flags almost everyone
            # and collapses precision (see tune.py).
            print(f"🔬 Running Optuna search for {args.model} "
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
            print(f"✅ Using Optuna-tuned hyperparameters: {best_params}")

        # === Cross-validated out-of-fold pass (stability + threshold) ===
        # One 5-fold CV pass over the train+val pool (early stopping inside each
        # fold). It gives both the threshold-free stability bands AND every row's
        # out-of-fold (OOF) probability, which is what the decision threshold is
        # chosen on — far more stable than reading one validation split.
        print("🔁 Running 5-fold CV (out-of-fold predictions, early stopping per fold)...")
        oof_proba, cv_stats = cross_val_oof(
            model, X_trainval, y_trainval, cv=5, calibrate=args.calibrate
        )
        for cv_metric, (mean, std) in cv_stats.items():
            mlflow.log_metric(f"cv_{cv_metric}_mean", mean)
            mlflow.log_metric(f"cv_{cv_metric}_std", std)
            print(f"   CV {cv_metric}: {mean:.3f} ± {std:.3f}")
        y_trainval_pos = y_trainval.reset_index(drop=True)

        # === Train Model and Track Training Time ===
        # Boosting models use early stopping on the validation split to pick the
        # tree count from data; RandomForest is fit as-is.
        t0 = time.time()
        fit_with_early_stopping(model, X_train, y_train, X_val, y_val)
        train_time = time.time() - t0
        if hasattr(model, "best_iteration") and model.best_iteration is not None:
            mlflow.log_metric("best_iteration", model.best_iteration)
            print(f"   Early stopping chose {model.best_iteration} trees")
        mlflow.log_metric("train_time", train_time)  # Track training performance
        print(f"✅ Model trained in {train_time:.2f} seconds")

        # === Probability calibration (fit on the validation split) ===
        # Class reweighting de-calibrates the raw scores; fit a post-hoc calibrator
        # on the held-out validation slice so the saved/evaluated model emits
        # trustworthy probabilities (needed for expected-value targeting). This is
        # the estimator used for every prediction below. Calibration is monotonic,
        # so PR-AUC/ROC-AUC and the OOF-selected threshold scale consistently.
        inference_model = calibrate_prefit(model, X_val, y_val, args.calibrate)
        if args.calibrate not in (None, "none"):
            print(f"📏 Calibrated final probabilities with {args.calibrate} "
                  f"(fit on the validation split)")

        # === STAGE 6: Select the decision threshold on OOF CV predictions ===
        # Either honor an explicit --threshold, or pick it from the out-of-fold
        # probabilities (every row predicted by a model that never saw it). Never
        # selected on the test set.
        if args.threshold is not None:
            threshold = args.threshold
            print(f"🎚️  Using fixed decision threshold: {threshold:.3f}")
        elif args.threshold_metric == "cost":
            threshold, neg_cost = select_threshold(
                y_trainval_pos, oof_proba, metric="cost", fn_fp_ratio=args.fn_fp_ratio
            )
            print(f"🎚️  Selected threshold {threshold:.3f} "
                  f"(CV-OOF cost={-neg_cost:.0f}, FN={args.fn_fp_ratio}xFP)")
        else:
            threshold, oof_score = select_threshold(
                y_trainval_pos, oof_proba, metric=args.threshold_metric
            )
            print(f"🎚️  Selected threshold {threshold:.3f} "
                  f"(CV-OOF {args.threshold_metric}={oof_score:.3f})")
        mlflow.log_param("selected_threshold", threshold)
        mlflow.log_param("fn_fp_ratio", args.fn_fp_ratio)
        mlflow.log_text(str(threshold), artifact_file="threshold.txt")

        # === Final evaluation on the untouched test split ===
        print("📊 Evaluating model performance on held-out test set...")
        t1 = time.time()
        proba = predict_positive_probability(inference_model, X_test)  # P(churn)
        pred_time = time.time() - t1
        mlflow.log_metric("pred_time", pred_time)

        ranking = threshold_free_metrics(y_test, proba)
        operating = metrics_at_threshold(y_test, proba, threshold)
        y_pred = (proba >= threshold).astype(int)

        precision = operating["precision"]  # Of predicted churners, how many churned?
        recall = operating["recall"]        # Of actual churners, how many we caught?
        f1 = operating["f1"]                # Harmonic mean of precision and recall
        roc_auc = ranking["roc_auc"]        # Threshold-independent ranking quality
        pr_auc = ranking["pr_auc"]          # PR-AUC — prevalence-aware ranking quality

        test_cost = expected_cost(operating, args.fn_fp_ratio)

        for metric_name, metric_value in {**ranking, **operating}.items():
            mlflow.log_metric(metric_name, metric_value)
        mlflow.log_metric("test_cost", test_cost)

        print(f"🎯 Model Performance (test):")
        print(f"   Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")
        print(f"   ROC AUC: {roc_auc:.3f} | PR AUC: {pr_auc:.3f}")
        print(f"   Confusion @ {threshold:.3f}: "
              f"TP={operating['tp']} FP={operating['fp']} "
              f"FN={operating['fn']} TN={operating['tn']}")
        print(f"   Business cost (FN={args.fn_fp_ratio}xFP): {test_cost:.0f} "
              f"FP-equivalents [{args.fn_fp_ratio}x{operating['fn']} + {operating['fp']}]")

        # === Probability calibration check ===
        # Measured on both the test set and (more stably) the OOF predictions —
        # both now reflect the post-hoc calibrator applied above (unless
        # --calibrate none), so this verifies the probabilities are actually
        # trustworthy rather than just flagging that they aren't. Calibration is a
        # monotonic rescaling, so it leaves the threshold decisions and PR-AUC
        # untouched while pulling Brier/ECE down.
        brier, ece, table = calibration_report(y_test, proba)
        brier_oof, ece_oof, _ = calibration_report(y_trainval_pos, oof_proba)
        mlflow.log_metric("brier", brier)
        mlflow.log_metric("ece", ece)
        mlflow.log_metric("brier_oof", brier_oof)
        mlflow.log_metric("ece_oof", ece_oof)
        mlflow.log_text(reliability_table_text(table), artifact_file="calibration_reliability_test.csv")
        print(f"📐 Calibration check (lower = better):")
        print(f"   Brier: {brier:.4f} (test) | {brier_oof:.4f} (OOF)")
        print(f"   ECE:   {ece:.4f} (test) | {ece_oof:.4f} (OOF)")
        if ece_oof > 0.05:
            hint = ("try --calibrate isotonic/sigmoid" if args.calibrate in (None, "none")
                    else "consider the other --calibrate method or more calibration data")
            print(f"   ⚠️  ECE > 0.05 — probabilities still not well calibrated ({hint}) "
                  "before using them as true probabilities.")

        # === Feature importance (permutation, on the test split) ===
        # Model-agnostic importance: how much test PR-AUC drops when each feature's
        # values are shuffled. Far more honest than a tree's split-count importance
        # (which inflates high-cardinality features), and it answers the other half
        # of a churn project — not just "who churns" but "which signals drive it".
        print("🔍 Computing permutation importance on the test split...")
        perm = permutation_importance(
            inference_model, X_test, y_test,
            scoring="average_precision", n_repeats=10, random_state=42, n_jobs=-1,
        )
        importance = (
            pd.DataFrame({
                "feature": feature_cols,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            })
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True)
        )
        importance_path = os.path.join(artifacts_dir, "feature_importance.csv")
        importance.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)
        print("   Top 10 features by permutation importance (PR-AUC drop):")
        print(importance.head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        # === STAGE 7: Model Serialization and Logging ===
        print("💾 Saving model to MLflow...")
        # Log the calibrated estimator actually used for inference (falls back to
        # the bare model when --calibrate none) so the saved artifact reproduces
        # the evaluated probabilities exactly.
        mlflow.sklearn.log_model(
            inference_model,
            artifact_path="model"  # This creates a 'model/' folder in MLflow run artifacts
        )
        print("✅ Model saved to MLflow")

        # === Final Performance Summary ===
        print(f"\n⏱️  Performance Summary:")
        print(f"   Training time: {train_time:.2f}s")
        print(f"   Inference time: {pred_time:.4f}s")
        print(f"   Samples per second: {len(X_test)/pred_time:.0f}")
        
        print(f"\n📈 Detailed Classification Report:")
        print(classification_report(y_test, y_pred, digits=3))


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
