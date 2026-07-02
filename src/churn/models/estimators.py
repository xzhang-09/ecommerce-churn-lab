"""Candidate estimator definitions for the churn pipeline.

The model zoo (RandomForest / LightGBM / XGBoost) and the helpers that select
one by name. Kept separate from the pipeline orchestration so adding or tuning a
candidate model is a one-file change.
"""

from dataclasses import dataclass

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


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
