import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBClassifier
import optuna
from optuna.samplers import TPESampler

# Objective tuned on a threshold-free metric (PR-AUC) so tuning is decoupled from
# the operating threshold, which is selected separately on a validation split.
TUNE_METRIC = "average_precision"
THRESHOLD_METRIC = "f1"
FEATURES_PATH = "data/processed/ecommerce_churn_features.csv"


def make_objective(X_train, y_train, scoring=TUNE_METRIC, cv_splits=5, seed=42):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
            "random_state": seed,
            "n_jobs": -1,
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "logloss",
        }
        model = XGBClassifier(**params)
        # StratifiedKFold is used automatically for a classification scorer.
        scores = cross_val_score(model, X_train, y_train, cv=cv_splits, scoring=scoring)
        return scores.mean()

    return objective


def select_threshold(y_true, proba, metric=THRESHOLD_METRIC):
    scorers = {"f1": f1_score, "precision": precision_score, "recall": recall_score}
    scorer = scorers[metric]
    best_t, best_v = 0.5, -1.0
    for t in np.unique(np.round(proba, 4)):
        v = scorer(y_true, (proba >= t).astype(int), zero_division=0)
        if v > best_v:
            best_v, best_t = v, float(t)
    return best_t


def main():
    print("=== Phase 2: Modeling with XGBoost ===")

    df = pd.read_csv(FEATURES_PATH)

    # target must be numeric 0/1
    if df["Churn"].dtype == "object":
        df["Churn"] = df["Churn"].str.strip().map({"No": 0, "Yes": 1})

    assert df["Churn"].isna().sum() == 0, "Churn has NaNs"
    assert set(df["Churn"].unique()) <= {0, 1}, "Churn not 0/1"

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Three-way split: train (fit) / val (threshold) / test (report once).
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=42
    )

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(make_objective(X_train, y_train), n_trials=30)
    print("Best Params:", study.best_params)
    print(f"Best CV {TUNE_METRIC}:", study.best_value)

    final_model = XGBClassifier(
        **study.best_params,
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric="logloss",
    )
    final_model.fit(X_train, y_train)

    # Choose the operating threshold on validation, never on test.
    val_proba = final_model.predict_proba(X_val)[:, 1]
    threshold = select_threshold(y_val, val_proba, THRESHOLD_METRIC)
    print(f"\nSelected threshold on validation ({THRESHOLD_METRIC}): {threshold:.3f}")

    test_proba = final_model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)
    print(f"Test ROC AUC: {roc_auc_score(y_test, test_proba):.3f} | "
          f"PR AUC: {average_precision_score(y_test, test_proba):.3f}")
    print("\nHoldout test report:")
    print(classification_report(y_test, test_pred, digits=3))


if __name__ == "__main__":
    main()
