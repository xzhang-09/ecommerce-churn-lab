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


def _map_binary_series(s: pd.Series) -> pd.Series:
    """
    Apply deterministic binary encoding to 2-category features.
    
    This function implements the core binary encoding logic that converts
    categorical features with exactly 2 values into 0/1 integers. The mappings
    are deterministic so repeated analysis runs produce the same encoded values.

    """
    mapping = _binary_mapping_for_series(s)
    if mapping is not None:
        return s.astype(str).map(mapping).astype("Int64")

    # === NON-BINARY FEATURES ===
    # Return unchanged - will be handled by one-hot encoding
    return s


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
            df[col] = df[col].astype(str).map(mapping).fillna(0).astype(int)

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


def build_features(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    Apply complete feature engineering pipeline for training data.
    
    This is the main feature engineering function that transforms raw customer data
    into ML-ready features for offline model training and evaluation.

    """
    df = df.copy()
    print(f"🔧 Starting feature engineering on {df.shape[1]} columns...")

    # === STEP 1: Identify Feature Types ===
    # Find categorical columns (object dtype) excluding the target variable
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != target_col]
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    
    print(f"   📊 Found {len(obj_cols)} categorical and {len(numeric_cols)} numeric columns")

    # === STEP 2: Split Categorical by Cardinality ===
    # Binary features (exactly 2 unique values) get binary encoding
    # Multi-category features (>2 unique values) get one-hot encoding
    binary_cols = [c for c in obj_cols if df[c].dropna().nunique() == 2]
    multi_cols = [c for c in obj_cols if df[c].dropna().nunique() > 2]
    
    print(f"   🔢 Binary features: {len(binary_cols)} | Multi-category features: {len(multi_cols)}")
    if binary_cols:
        print(f"      Binary: {binary_cols}")
    if multi_cols:
        print(f"      Multi-category: {multi_cols}")

    # === STEP 3: Fit and apply reusable schema ===
    schema = build_feature_schema(df, target_col=target_col)
    df = apply_feature_schema(df, schema)

    print(f"✅ Feature engineering complete: {df.shape[1]} final features")
    return df
