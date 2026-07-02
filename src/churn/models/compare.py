"""Fair, cross-validated comparison of the candidate models."""

import logging
import time

import mlflow
import pandas as pd

from churn.models.estimators import get_model_specs
from churn.models.evaluate import expected_cost, metrics_at_threshold, select_threshold
from churn.models.training import cross_val_oof

logger = logging.getLogger(__name__)


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
        logger.info(f"🔎 Cross-validating candidate model: {spec.name}")
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
    logger.info(f"\n📊 Model comparison — ranked by CV PR-AUC (mean ± std over folds); "
                f"precision/recall/f1 and oof_cost (FN={fn_fp_ratio}xFP) are out-of-fold "
                f"context at each model's threshold:")
    logger.info(results.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    best = results.iloc[0]
    logger.info(f"\n🏆 Best by CV PR-AUC: {best['model']} "
                f"({best['cv_pr_auc']:.3f} ± {best['cv_pr_std']:.3f})")
    return results
