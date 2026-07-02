"""Metrics, threshold selection, and calibration diagnostics.

Pure evaluation functions with no MLflow or pipeline dependencies, so they are
easy to unit-test and reuse. Two families:

* *ranking* / threshold-free quality (ROC-AUC, PR-AUC);
* *operating-point* quality at a chosen decision threshold (precision/recall/f1,
  confusion counts, the FN/FP business cost), plus the threshold search itself
  and calibration (Brier / ECE) reporting.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


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


def select_threshold(y_true, probabilities, metric: str = "cost", fn_fp_ratio: float = 10.0):
    """Pick the decision threshold that optimizes `metric` on the given
    (validation) data. `metric="cost"` minimizes the FN/FP business cost
    (default, FN is `fn_fp_ratio`x an FP); "f1"/"precision"/"recall" maximize
    that metric instead. The threshold must be chosen on held-out data that is
    not the test set — picking it on the test set is leakage and inflates the
    reported score. Returns ``(threshold, best_objective_value)`` where the
    objective is the quantity being maximized (negative cost for "cost").

    Vectorized: for every candidate cut point the whole confusion matrix is
    derived at once from cumulative label counts, so the scan is O(n log n)
    instead of the previous O(n^2) (one full confusion-matrix pass per
    candidate). On ties the smallest threshold wins, matching the old
    ascending-scan behavior.
    """
    proba = np.round(np.asarray(probabilities, dtype=float), 4)
    y = np.asarray(y_true, dtype=int)
    # Candidate thresholds = the distinct predicted probabilities; every
    # achievable confusion matrix is realized at one of these cut points.
    thresholds = np.unique(proba)
    if thresholds.size == 0:
        return 0.5, -np.inf

    # Sort once; a prediction is positive iff proba >= t, so counting labels in
    # the sorted right-hand tail from each threshold gives tp/fp directly.
    order = np.argsort(proba, kind="mergesort")
    proba_sorted = proba[order]
    y_sorted = y[order]
    pos_from_right = np.cumsum(y_sorted[::-1])[::-1]   # positives with proba >= tail start
    cnt_from_right = np.arange(len(proba), 0, -1)       # rows with proba >= tail start

    # Every threshold is an actual probability value, so this index is always in range.
    idx = np.searchsorted(proba_sorted, thresholds, side="left")
    tp = pos_from_right[idx]
    predicted_positive = cnt_from_right[idx]
    fp = predicted_positive - tp
    total_positive = int(y.sum())
    fn = total_positive - tp

    if metric == "cost":
        objective = -(fn_fp_ratio * fn + fp).astype(float)
    elif metric == "precision":
        denom = tp + fp
        objective = np.divide(tp, denom, out=np.zeros(len(thresholds)), where=denom > 0)
    elif metric == "recall":
        objective = (tp / total_positive) if total_positive else np.zeros(len(thresholds))
    elif metric == "f1":
        denom = 2 * tp + fp + fn
        objective = np.divide(2 * tp, denom, out=np.zeros(len(thresholds)), where=denom > 0)
    else:
        raise ValueError(f"Unknown threshold metric '{metric}'.")

    best = int(np.argmax(objective))  # first max -> smallest threshold on ties
    return float(thresholds[best]), float(objective[best])
