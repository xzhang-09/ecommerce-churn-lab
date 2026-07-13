import pandas as pd

from churn.models.report import render_results_markdown


def _sample_results():
    return {
        "model": "lightgbm",
        "run_id": "abc123",
        "git_commit": "deadbee",
        "timestamp": "2026-07-07 12:00 UTC",
        "dataset": {"raw_rows": 5630, "clean_rows": 5073,
                    "churn_rate": 0.166, "target": "Churn"},
        "test_rows": 1015, "test_pos": 168, "test_neg": 847,
        "threshold": 0.114, "fn_fp_ratio": 10.0,
        "ranking": {"pr_auc": 0.931, "roc_auc": 0.984},
        "operating": {"precision": 0.671, "recall": 0.982, "f1": 0.797,
                      "tp": 165, "fp": 81, "fn": 3, "tn": 766},
        "test_cost": 111.0,
        "brier": 0.0317, "ece": 0.0191, "brier_oof": 0.04, "ece_oof": 0.03,
        "comparison": pd.DataFrame(
            {"model": ["lightgbm", "xgboost"],
             "cv_pr_auc": [0.897, 0.892], "cv_pr_std": [0.022, 0.018]}
        ),
        "importance": pd.DataFrame(
            {"feature": ["Tenure", "Complain"],
             "importance_mean": [0.531, 0.196],
             "importance_std": [0.01, 0.01]}
        ),
    }


def test_render_includes_core_metrics_and_provenance():
    md = render_results_markdown(_sample_results())
    # Provenance + headline metrics land in the report.
    assert "abc123" in md
    assert "deadbee" in md
    assert "16.6%" in md
    assert "0.931" in md  # PR-AUC
    assert "TP=165, FP=81, FN=3, TN=766" in md
    assert "111 FP-equivalents" in md
    # Threshold-labelled operating metrics.
    assert "@ threshold 0.114" in md
    # Optional sections render their tables.
    assert "Cross-validated model comparison" in md
    assert "`Tenure`" in md
    # It is explicitly marked as generated so nobody hand-edits it.
    assert "generated" in md.lower()


def test_render_tolerates_missing_optional_sections():
    minimal = {
        "model": "lightgbm",
        "ranking": {"pr_auc": 0.9},
        "operating": {"precision": 0.6, "recall": 0.9, "f1": 0.72},
        "comparison": None,
        "importance": None,
    }
    md = render_results_markdown(minimal)
    assert "0.900" in md
    assert "Cross-validated model comparison" not in md
    assert "permutation-importance" not in md.lower()
