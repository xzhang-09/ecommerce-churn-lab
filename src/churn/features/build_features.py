import warnings

import pandas as pd


def _binary_mapping_for_series(s: pd.Series) -> dict | None:
    vals = list(pd.Series(s.dropna().unique()).astype(str))
    valset = set(vals)

    if valset == {"Yes", "No"}:
        return {"No": 0, "Yes": 1}

    if valset == {"Male", "Female"}:
        return {"Female": 0, "Male": 1}

    if len(vals) == 2:
        sorted_vals = sorted(vals)
        return {sorted_vals[0]: 0, sorted_vals[1]: 1}

    return None


def build_feature_schema(df: pd.DataFrame, target_col: str = "Churn") -> dict:
    """Fit reusable categorical encoding metadata from a training dataframe."""
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != target_col]
    binary_mappings = {
        c: _binary_mapping_for_series(df[c])
        for c in obj_cols
        if df[c].dropna().nunique() == 2
    }
    binary_mappings = {c: m for c, m in binary_mappings.items() if m is not None}
    multi_categories = {
        c: sorted(pd.Series(df[c].dropna().unique()).astype(str).tolist())
        for c in obj_cols
        if df[c].dropna().nunique() > 2
    }

    encoded = apply_feature_schema(
        df,
        {
            "target_col": target_col,
            "binary_mappings": binary_mappings,
            "multi_categories": multi_categories,
            "feature_columns": None,
        },
    )
    feature_columns = [c for c in encoded.columns if c != target_col]
    return {
        "target_col": target_col,
        "binary_mappings": binary_mappings,
        "multi_categories": multi_categories,
        "feature_columns": feature_columns,
    }


def apply_feature_schema(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Apply fitted feature metadata so new data keeps the training columns."""
    df = df.copy()
    target_col = schema.get("target_col", "Churn")
    binary_mappings = schema.get("binary_mappings", {})
    multi_categories = schema.get("multi_categories", {})

    for col, mapping in binary_mappings.items():
        if col not in df.columns:
            df[col] = 0
        else:
            col_str = df[col].astype(str)
            mapped = col_str.map(mapping)
            # A non-null value that isn't a mapping key would be silently coerced
            # to 0 by the fillna below, quietly turning it into the negative class.
            # That masks upstream schema drift (e.g. "Yes"/"No" becoming "Y"/"N"),
            # so surface it instead of letting the column degrade unnoticed.
            unseen = sorted(set(col_str[mapped.isna() & df[col].notna()].unique()))
            if unseen:
                warnings.warn(
                    f"Column '{col}': values {unseen} were not seen when the feature "
                    f"schema was fit (known: {sorted(mapping)}); encoding them as 0. "
                    "This usually means the input schema changed upstream.",
                    stacklevel=2,
                )
            df[col] = mapped.fillna(0).astype(int)

    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)

    for col, categories in multi_categories.items():
        if col not in df.columns:
            df[col] = pd.Series([None] * len(df), index=df.index)
        df[col] = pd.Categorical(df[col].astype(str), categories=categories)

    if multi_categories:
        df = pd.get_dummies(df, columns=list(multi_categories), drop_first=True)

    for col in binary_mappings:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    feature_columns = schema.get("feature_columns")
    if feature_columns is not None:
        ordered_columns = list(feature_columns)
        if target_col in df.columns:
            ordered_columns = [target_col] + ordered_columns
        df = df.reindex(columns=ordered_columns, fill_value=0)

    return df
