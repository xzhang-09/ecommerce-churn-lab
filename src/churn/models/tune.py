import logging

import optuna
from optuna.samplers import TPESampler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


def _suggest_params(trial, model_type: str):
    """Per-model Optuna search space. Both spaces include explicit regularization
    because the dataset is small (~5k rows after dedup) and over-fitting control
    matters more than chasing raw capacity."""
    if model_type == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
    if model_type == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            # num_leaves is LightGBM's primary over-fit knob (leaf-wise growth);
            # allow it to go small for this small dataset.
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            # subsample (bagging_fraction) only takes effect when subsample_freq >= 1.
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
    raise ValueError(f"Unsupported model_type for tuning: {model_type!r} (use 'xgboost' or 'lightgbm')")


def _build_estimator(model_type: str, params: dict, scale_pos_weight: float, random_state: int):
    """Construct an estimator from tuned params, matching each model's imbalance
    convention: XGBoost uses scale_pos_weight, LightGBM uses class_weight."""
    common = dict(n_jobs=-1, random_state=random_state)
    if model_type == "xgboost":
        return XGBClassifier(**params, **common, eval_metric="logloss",
                             scale_pos_weight=scale_pos_weight)
    if model_type == "lightgbm":
        return LGBMClassifier(**params, **common, class_weight="balanced", verbosity=-1)
    raise ValueError(f"Unsupported model_type for tuning: {model_type!r} (use 'xgboost' or 'lightgbm')")


def build_tuned_estimator(model_type: str, best_params: dict, scale_pos_weight: float,
                          random_state: int = 42):
    """Public helper so the pipeline can rebuild the tuned estimator without
    duplicating the per-model construction logic."""
    return _build_estimator(model_type, best_params, scale_pos_weight, random_state)


def tune_model(X, y, model_type: str = "xgboost", scoring: str = "average_precision",
               n_trials: int = 20, cv: int = 5, random_state: int = 42):
    """
    Tunes an XGBoost or LightGBM model with Optuna.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
        model_type: "xgboost" or "lightgbm". Selects both the search space and the
            estimator's imbalance handling (scale_pos_weight vs class_weight).
        scoring: sklearn scoring metric to optimize. Defaults to
            "average_precision" (PR-AUC) — a threshold-free ranking metric suited
            to imbalanced churn and decoupled from any decision threshold (which
            is selected separately, on out-of-fold predictions, after tuning).
            Avoid "recall" alone — it collapses precision by flagging everyone.
        n_trials: number of Optuna trials to run.
        cv: number of stratified CV folds (StratifiedKFold is used automatically
            for classification scorers).
        random_state: seed for both the sampler and the estimator so tuning is
            reproducible.
    """
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()

    def objective(trial):
        params = _suggest_params(trial, model_type)
        model = _build_estimator(model_type, params, scale_pos_weight, random_state)
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        return scores.mean()

    # Seed the sampler so repeated runs reproduce the same search trajectory.
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials)

    logger.info(f"Best Params ({model_type}): {study.best_params}")
    logger.info(f"Best {scoring}: {study.best_value}")
    return study.best_params
