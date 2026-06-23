import pandas as pd


def preprocess_data(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    Basic cleaning for the e-commerce churn dataset.
    - trim column names
    - drop obvious ID cols
    - map target Churn to 0/1 if needed
    - simple NA handling
    - drop exact duplicate rows
    """
    # tidy headers
    df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace

    # drop ids if present
    for col in ["CustomerID", "customerID", "customer_id"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # target to 0/1 if it's Yes/No
    if target_col in df.columns and df[target_col].dtype == "object":
        df[target_col] = df[target_col].str.strip().map({"No": 0, "Yes": 1})

    # Collapse duplicate-meaning category labels (raw data has both spellings
    # for the same thing, e.g. "Phone" and "Mobile Phone"). Must happen before
    # dedup below — otherwise rows that are identical except for one of these
    # label variants don't get recognized as duplicates.
    category_aliases = {
        "PreferredLoginDevice": {"Phone": "Mobile Phone"},
        "PreferredPaymentMode": {"COD": "Cash on Delivery", "CC": "Credit Card"},
        "PreferedOrderCat": {"Mobile": "Mobile Phone"},
    }
    for col, aliases in category_aliases.items():
        if col in df.columns:
            df[col] = df[col].replace(aliases)

    # Missing-value handling — record, don't silently fill with 0.
    # Numeric NA in this dataset is informative but ambiguous, and a blanket
    # fillna(0) conflates "unknown" with a real value whose meaning differs per
    # column (Tenure=0 -> brand-new customer, DaySinceLastOrder=0 -> ordered
    # today, OrderCount=0 -> never ordered). That injects misleading signal.
    # Instead we (1) flag missingness explicitly with a binary indicator column,
    # then (2) leave the NaNs for median imputation. The imputation itself is
    # deferred to impute_numeric() so it can be fit on the training split only
    # (median computed from test rows would be leakage); get_dummies downstream
    # ignores NaN safely for the non-numeric columns.
    num_cols = df.select_dtypes(include=["number"]).columns
    missing_cols = [c for c in num_cols if c != target_col and df[c].isna().any()]
    for c in missing_cols:
        df[f"{c}_missing"] = df[c].isna().astype(int)

    # CRITICAL: with CustomerID dropped, ~10% of rows in the raw e-commerce
    # dataset are exact duplicates of another row. Left in place, train_test_split
    # randomly puts copies of the same row on both sides of the split, so the
    # model partly memorizes test rows instead of generalizing — this is why an
    # untuned XGBoost was hitting recall=1.0 / ROC AUC=0.998 (17% of the test
    # set turned out to be duplicates of training rows). Must dedupe before
    # the split, not after, or this leakage comes right back.
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"🧹 Dropped {n_dropped} exact duplicate rows ({n_dropped / n_before:.1%}) to prevent train/test leakage")

    return df


def impute_numeric(df: pd.DataFrame, medians: "pd.Series | None" = None):
    """
    Median-impute numeric columns, leakage-free.

    Pass `medians` (a Series fit on the training split) to apply the *training*
    medians to validation/test data; otherwise medians are computed from `df`
    itself (acceptable only for offline, single-table artifacts where no split
    exists). Returns ``(imputed_df, medians)`` so the caller can reuse the fitted
    medians on later splits.

    Pairs with the ``*_missing`` indicator columns added in ``preprocess_data``:
    the model still sees that a value was originally absent even after the gap is
    filled with a neutral central value.
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=["number"]).columns
    if medians is None:
        medians = df[num_cols].median()
    df[num_cols] = df[num_cols].fillna(medians)
    # Guard against a column that was entirely NaN (median undefined) -> fill 0.
    df[num_cols] = df[num_cols].fillna(0)
    return df, medians
