import pandas as pd

from src.features.build_features import apply_feature_schema, build_feature_schema


def test_feature_schema_reuses_training_columns_for_unseen_categories():
    train = pd.DataFrame(
        {
            "Churn": [0, 1, 0, 1],
            "Gender": ["Female", "Male", "Female", "Male"],
            "PreferredPaymentMode": ["Credit Card", "Debit Card", "UPI", "Debit Card"],
            "Tenure": [1.0, 2.0, 3.0, 4.0],
        }
    )
    incoming = pd.DataFrame(
        {
            "Churn": [0, 1],
            "Gender": ["Male", "Female"],
            "PreferredPaymentMode": ["Crypto", "Credit Card"],
            "Tenure": [5.0, 6.0],
        }
    )

    schema = build_feature_schema(train, target_col="Churn")
    train_features = apply_feature_schema(train, schema)
    incoming_features = apply_feature_schema(incoming, schema)

    assert list(incoming_features.columns) == list(train_features.columns)
    assert "PreferredPaymentMode_Crypto" not in incoming_features.columns
    assert incoming_features.loc[0, "PreferredPaymentMode_Debit Card"] == 0
    assert incoming_features.loc[0, "PreferredPaymentMode_UPI"] == 0
