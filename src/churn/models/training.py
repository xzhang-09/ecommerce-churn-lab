"""Model fitting helpers: probability extraction, calibration, early stopping,
and the cross-validated out-of-fold (OOF) pass.

These are the reusable "how to fit and score one estimator honestly" building
blocks shared by the final-model path and the model comparison.
"""

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split


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
