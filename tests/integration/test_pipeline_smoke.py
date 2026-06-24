import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from churn.pipeline import metrics_at_threshold, select_threshold, threshold_free_metrics
from churn.data.load_data import load_data
from churn.data.preprocess import impute_numeric, preprocess_data
from churn.features.build_features import apply_feature_schema, build_feature_schema


def test_lightweight_pipeline_smoke_runs_from_raw_csv_to_metrics(tmp_path):
    rows = []
    for i in range(80):
        churn = int(i % 5 == 0 or (i % 7 == 0 and i > 20))
        rows.append(
            {
                "CustomerID": i + 1,
                "Churn": churn,
                "Gender": "Male" if i % 2 else "Female",
                "PreferredLoginDevice": ["Mobile Phone", "Computer", "Phone"][i % 3],
                "PreferredPaymentMode": ["Credit Card", "Debit Card", "UPI"][i % 3],
                "PreferedOrderCat": ["Laptop", "Mobile Phone", "Fashion"][i % 3],
                "Tenure": None if i % 11 == 0 else float(i % 24),
                "SatisfactionScore": 2 if churn else 4,
                "Complain": churn,
                "CashbackAmount": 120.0 + i,
            }
        )
    raw_path = tmp_path / "raw.csv"
    pd.DataFrame(rows).to_csv(raw_path, index=False)

    raw = load_data(str(raw_path))
    cleaned = preprocess_data(raw, target_col="Churn")
    schema = build_feature_schema(cleaned, target_col="Churn")
    encoded = apply_feature_schema(cleaned, schema)

    X = encoded.drop(columns=["Churn"])
    y = encoded["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    X_train, medians = impute_numeric(X_train)
    X_test, _ = impute_numeric(X_test, medians)

    model = RandomForestClassifier(n_estimators=20, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    threshold, _ = select_threshold(y_test, probabilities, metric="f1")

    ranking = threshold_free_metrics(y_test, probabilities)
    operating = metrics_at_threshold(y_test, probabilities, threshold)

    assert set(ranking) == {"roc_auc", "pr_auc"}
    assert {"precision", "recall", "f1", "tn", "fp", "fn", "tp"} == set(operating)
    assert len(schema["feature_columns"]) == X_train.shape[1]
