"""End-to-end test that actually runs ``pipeline.main`` on synthetic data.

The unit tests exercise individual helpers but never the ~300-line ``main``
orchestrator (MLflow logging, artifact writes, split wiring). This runs the
whole thing on a small synthetic dataset in an isolated working directory, so a
regression anywhere in the orchestration is caught instead of only surfacing on
a real multi-minute run.
"""

import joblib
import pandas as pd

from churn import pipeline


def _make_synthetic_raw(n: int = 180) -> pd.DataFrame:
    rows = []
    for i in range(n):
        # A learnable signal plus noise so both classes are well populated.
        base = (i % 3 == 0) or (i % 7 == 0)
        churn = int(base ^ (i % 11 == 0))
        rows.append(
            {
                "CustomerID": i + 1,
                "Churn": churn,
                "Gender": "Male" if i % 2 else "Female",
                "PreferredLoginDevice": ["Mobile Phone", "Computer", "Phone"][i % 3],
                "PreferredPaymentMode": ["Credit Card", "Debit Card", "UPI"][i % 3],
                "PreferedOrderCat": ["Laptop", "Mobile Phone", "Fashion"][i % 3],
                "Tenure": None if i % 13 == 0 else float(i % 24),
                "SatisfactionScore": 2 if churn else 4,
                "Complain": churn,
                "CashbackAmount": 120.0 + (i % 50),
            }
        )
    return pd.DataFrame(rows)


def test_pipeline_main_runs_end_to_end_and_writes_artifacts(tmp_path, monkeypatch):
    raw_path = tmp_path / "raw.csv"
    _make_synthetic_raw().to_csv(raw_path, index=False)

    # main() writes data/processed, artifacts, and mlruns relative to cwd.
    monkeypatch.chdir(tmp_path)

    args = pipeline.build_arg_parser().parse_args(
        [
            "--input", str(raw_path),
            "--target", "Churn",
            "--skip_validation",
            "--model", "random_forest",
            "--calibrate", "none",
            "--mlflow_uri", f"file://{tmp_path}/mlruns",
        ]
    )

    pipeline.main(args)

    # Core artifacts exist and the preprocessing bundle is complete + provenanced.
    assert (tmp_path / "data" / "processed" / "ecommerce_churn_cleaned.csv").exists()
    assert (tmp_path / "artifacts" / "feature_columns.json").exists()
    assert (tmp_path / "artifacts" / "feature_importance.csv").exists()

    bundle_path = tmp_path / "artifacts" / "preprocessing.pkl"
    assert bundle_path.exists()
    bundle = joblib.load(bundle_path)
    assert set(bundle) >= {"feature_columns", "feature_schema", "target", "medians", "mlflow_run_id"}
    assert bundle["mlflow_run_id"]
