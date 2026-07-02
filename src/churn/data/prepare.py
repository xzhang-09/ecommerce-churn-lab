import logging
import os

from churn.data.load_data import load_data
from churn.data.preprocess import preprocess_data, impute_numeric
from churn.features.build_features import apply_feature_schema, build_feature_schema
from churn.logging_utils import configure_logging

logger = logging.getLogger(__name__)

RAW = "data/raw/E Commerce Dataset.xlsx"
CLEANED_OUT = "data/processed/ecommerce_churn_cleaned.csv"
FEATURES_OUT = "data/processed/ecommerce_churn_features.csv"


def main(raw_path: str = RAW, cleaned_out: str = CLEANED_OUT, features_out: str = FEATURES_OUT) -> None:
    configure_logging()
    # 1) load raw
    df = load_data(raw_path)

    # 2) preprocess (adds *_missing indicator columns, leaves numeric NaNs)
    df = preprocess_data(df, target_col="Churn")

    # 2b) median-impute the remaining numeric gaps. This is a single offline
    # table with no train/test split, so full-data medians are acceptable here;
    # the modeling pipeline imputes with train-only medians to stay leakage-free.
    df, _ = impute_numeric(df)

    # 3) ensure target is 0/1 only if still object
    if "Churn" in df.columns and df["Churn"].dtype == "object":
        df["Churn"] = df["Churn"].str.strip().map({"No": 0, "Yes": 1}).astype("Int64")

    # sanity checks
    assert df["Churn"].isna().sum() == 0, "Churn has NaNs after preprocess"
    assert set(df["Churn"].unique()) <= {0, 1}, "Churn not 0/1 after preprocess"

    # 4) save cleaned data separately from model-ready features
    os.makedirs(os.path.dirname(cleaned_out), exist_ok=True)
    df.to_csv(cleaned_out, index=False)
    logger.info(f"✅ Cleaned dataset saved to {cleaned_out} | Shape: {df.shape}")

    # 5) features
    feature_schema = build_feature_schema(df, target_col="Churn")
    df_features = apply_feature_schema(df, feature_schema)
    os.makedirs(os.path.dirname(features_out), exist_ok=True)
    df_features.to_csv(features_out, index=False)
    logger.info(f"✅ Feature dataset saved to {features_out} | Shape: {df_features.shape}")


if __name__ == "__main__":
    main()
