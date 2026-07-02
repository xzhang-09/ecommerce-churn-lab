import warnings

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from churn.data.preprocess import impute_numeric, preprocess_data
from churn.features.build_features import apply_feature_schema, build_feature_schema
from churn.models.score import score_dataframe


def _train_tiny_model():
    raw = pd.DataFrame(
        {
            "CustomerID": range(1, 13),
            "Churn": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "Gender": ["Male", "Female"] * 6,
            "PreferredPaymentMode": ["Credit Card", "Debit Card", "UPI"] * 4,
            "Tenure": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        }
    )
    cleaned = preprocess_data(raw, target_col="Churn")
    schema = build_feature_schema(cleaned, target_col="Churn")
    encoded = apply_feature_schema(cleaned, schema)
    X = encoded.drop(columns=["Churn"])
    y = encoded["Churn"]
    X, medians = impute_numeric(X)
    model = RandomForestClassifier(n_estimators=20, random_state=42).fit(X, y)
    return model, schema, medians


def test_score_dataframe_returns_one_ranked_row_per_customer():
    model, schema, medians = _train_tiny_model()

    # A fresh batch with no target column, one unseen payment mode, and a NaN.
    batch = pd.DataFrame(
        {
            "CustomerID": [101, 102, 103],
            "Gender": ["Female", "Male", "Female"],
            "PreferredPaymentMode": ["Credit Card", "Crypto", "UPI"],
            "Tenure": [4.0, None, 9.0],
        }
    )

    scored = score_dataframe(batch, model, schema, medians, threshold=0.5)

    assert len(scored) == len(batch)
    assert list(scored.columns) == ["CustomerID", "churn_probability", "churn_prediction"]
    assert set(scored["CustomerID"]) == {101, 102, 103}
    # Probabilities are valid and returned high-risk first.
    assert scored["churn_probability"].between(0, 1).all()
    assert scored["churn_probability"].is_monotonic_decreasing
    assert set(scored["churn_prediction"].unique()) <= {0, 1}


def test_score_dataframe_warns_on_unseen_binary_value():
    model, schema, medians = _train_tiny_model()
    batch = pd.DataFrame(
        {
            "CustomerID": [201],
            "Gender": ["Nonbinary"],  # unseen in training -> encoded 0 with a warning
            "PreferredPaymentMode": ["Credit Card"],
            "Tenure": [3.0],
        }
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        score_dataframe(batch, model, schema, medians, threshold=0.5)

    assert any("Gender" in str(w.message) for w in caught)
