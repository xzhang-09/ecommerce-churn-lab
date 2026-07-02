#!/usr/bin/env python3
"""Batch scoring: apply a trained churn model to a fresh batch of customers.

This closes the loop the modeling pipeline leaves open — turning "who is likely
to churn" into an actionable, ranked contact list. Everything needed to score
(the calibrated model, the leakage-free preprocessing state, and the selected
decision threshold) is loaded from a *single* MLflow run, so the three pieces
can never be silently mismatched (a model from one run with a threshold from
another). See :func:`load_run_artifacts`.

The transform mirrors training exactly: category-alias normalization and
missing-value indicators (``preprocess_data``), the fitted categorical
``feature_schema`` (``apply_feature_schema``), and the *training* imputation
medians (``impute_numeric``) — never medians recomputed from the new batch,
which would reintroduce train/serve skew.
"""

import argparse
import logging
import os

import joblib
import numpy as np
import pandas as pd

from churn.data.load_data import load_data
from churn.data.preprocess import impute_numeric, preprocess_data
from churn.features.build_features import apply_feature_schema
from churn.logging_utils import configure_logging

logger = logging.getLogger(__name__)

DEFAULT_ID_COLUMNS = ["CustomerID", "customerID", "customer_id"]


def _positive_probability(model, X):
    """P(churn) from whatever the loaded estimator exposes (mirrors the pipeline)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return 1 / (1 + np.exp(-model.decision_function(X)))
    return model.predict(X)


def score_dataframe(
    df: pd.DataFrame,
    model,
    feature_schema: dict,
    medians,
    threshold: float,
    id_column: str | None = None,
) -> pd.DataFrame:
    """Score a raw customer batch and return a ranked contact list.

    `df` is raw (pre-cleaning) input in the same shape as the training source.
    `medians` is the training-fit imputation medians (Series or column->value
    dict). Returns one row per input customer with the churn probability, the
    threshold decision, and the customer id when available, sorted by descending
    probability so the highest-risk customers are at the top.
    """
    target = feature_schema.get("target_col", "Churn")
    feature_columns = feature_schema["feature_columns"]

    # Capture the id before preprocessing drops it, so the output stays joinable.
    if id_column is None:
        id_column = next((c for c in DEFAULT_ID_COLUMNS if c in df.columns), None)
    ids = df[id_column].reset_index(drop=True) if id_column and id_column in df.columns else None

    # drop_duplicates=False: scoring must keep every customer row in the output.
    cleaned = preprocess_data(df, target_col=target, drop_duplicates=False)
    encoded = apply_feature_schema(cleaned, feature_schema)

    X = encoded.reindex(columns=feature_columns, fill_value=0)
    if not isinstance(medians, pd.Series):
        medians = pd.Series(medians)
    X, _ = impute_numeric(X, medians)

    proba = _positive_probability(model, X)
    result = pd.DataFrame(
        {
            "churn_probability": proba,
            "churn_prediction": (proba >= threshold).astype(int),
        }
    )
    if ids is not None:
        result.insert(0, id_column, ids)
    return result.sort_values("churn_probability", ascending=False).reset_index(drop=True)


def load_run_artifacts(run_id: str | None = None, tracking_uri: str | None = None):
    """Load (model, feature_schema, medians, threshold) from one MLflow run.

    When `run_id` is None the most recent finished run in the default experiment
    is used. Loading all four from the same run is what guarantees the model, its
    preprocessing state, and its threshold are a matched set.
    """
    import mlflow
    import mlflow.sklearn

    project_root = os.getcwd()
    mlflow.set_tracking_uri(tracking_uri or f"file://{project_root}/mlruns")

    if run_id is None:
        runs = mlflow.search_runs(
            experiment_names=["E-Commerce Churn"],
            order_by=["start_time DESC"],
            max_results=1,
        )
        if runs.empty:
            raise ValueError(
                "No MLflow runs found to score from. Train a model first with "
                "`churn-pipeline` / `python scripts/run_pipeline.py`, or pass --run-id."
            )
        run_id = runs.iloc[0]["run_id"]

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    client = mlflow.tracking.MlflowClient()
    local_dir = client.download_artifacts(run_id, "preprocessing.pkl")
    preprocessing = joblib.load(local_dir)
    threshold_path = client.download_artifacts(run_id, "threshold.txt")
    with open(threshold_path) as f:
        threshold = float(f.read().strip())

    feature_schema = preprocessing["feature_schema"]
    medians = preprocessing["medians"]
    return model, feature_schema, medians, threshold, run_id


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Score a batch of customers with a trained churn model."
    )
    parser.add_argument("--input", required=True,
                        help="path to the raw customer batch (CSV or Excel)")
    parser.add_argument("--output", default="data/processed/churn_scores.csv",
                        help="where to write the ranked scored output (default: "
                             "data/processed/churn_scores.csv)")
    parser.add_argument("--run-id", default=None,
                        help="MLflow run to load the model/preprocessing/threshold from "
                             "(default: the most recent run in the 'E-Commerce Churn' experiment)")
    parser.add_argument("--id-column", default=None,
                        help="customer id column to carry into the output "
                             "(default: auto-detect CustomerID)")
    parser.add_argument("--mlflow-uri", default=None,
                        help="override MLflow tracking URI (default: project_root/mlruns)")
    args = parser.parse_args(argv)
    configure_logging()

    logger.info("🔄 Loading trained model and preprocessing state from MLflow...")
    model, feature_schema, medians, threshold, run_id = load_run_artifacts(
        run_id=args.run_id, tracking_uri=args.mlflow_uri
    )
    logger.info(f"✅ Loaded run {run_id} (threshold={threshold:.3f})")

    df = load_data(args.input)
    logger.info(f"✅ Scoring {len(df)} customers...")
    scored = score_dataframe(
        df, model, feature_schema, medians, threshold, id_column=args.id_column
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    scored.to_csv(args.output, index=False)
    flagged = int(scored["churn_prediction"].sum())
    logger.info(f"✅ Wrote {len(scored)} scores to {args.output} "
                f"({flagged} flagged for retention outreach at threshold {threshold:.3f})")


if __name__ == "__main__":
    main()
