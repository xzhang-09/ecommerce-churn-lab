import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)


# === Validation rules (edit/version these in one place) ===
# Pulled out of the function body so the business constraints the model assumes
# are a self-documenting, reviewable block rather than magic numbers scattered
# through the suite. Categorical value sets, allowed numeric ranges, and the
# `mostly` tolerances all live here.
REQUIRED_COLUMNS = [
    "CustomerID", "Gender", "MaritalStatus", "Tenure",
    "PreferredLoginDevice", "PreferredPaymentMode", "SatisfactionScore", "Complain",
]
CATEGORICAL_VALUE_SETS = {
    "Gender": ["Male", "Female"],
    "MaritalStatus": ["Single", "Married", "Divorced"],
    # Raw data carries both "Phone" and "Mobile Phone"; preprocessing collapses
    # them later, so validation (which runs on raw data) accepts both spellings.
    "PreferredLoginDevice": ["Mobile Phone", "Phone", "Computer"],
    "CityTier": [1, 2, 3],
    "Complain": [0, 1],
}
NUMERIC_RANGES = {
    # column: (min_value, max_value, mostly)  — None means unbounded on that side.
    "Tenure": (0, 120, 0.99),            # months on the platform, can't be negative
    "SatisfactionScore": (1, 5, None),   # 1-5 rating scale
    "WarehouseToHome": (0, None, 0.99),  # distance must be positive
    "CashbackAmount": (0, None, None),   # cashback can't be negative
}
NOT_NULL_COLUMNS = ["CustomerID", "Gender", "SatisfactionScore"]


def validate_ecommerce_data(df) -> Tuple[bool, List[str]]:
    """
    Data quality validation for the E-Commerce Customer Churn dataset using
    Great Expectations. Validates schema, business logic constraints, and
    statistical properties that the ML model expects before training.

    Uses GX's modern (1.x) fluent API: a throwaway in-memory ("ephemeral")
    context wraps the dataframe in a batch, and each check is an
    Expectation object validated against that batch. The legacy
    `ge.dataset.PandasDataset` one-liner API used in older GX versions was
    removed in 1.x.
    """
    import great_expectations as gx

    logger.info("🔍 Starting data validation with Great Expectations...")

    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas("ecommerce_pandas_datasource")
    data_asset = data_source.add_dataframe_asset(name="ecommerce_data")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("batch_definition")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    expectations = []

    # === SCHEMA VALIDATION - ESSENTIAL COLUMNS ===
    logger.info("   📋 Validating schema and required columns...")
    for column in REQUIRED_COLUMNS:
        expectations.append(gx.expectations.ExpectColumnToExist(column=column))

    # === BUSINESS LOGIC VALIDATION ===
    logger.info("   💼 Validating business logic constraints...")
    for column, value_set in CATEGORICAL_VALUE_SETS.items():
        expectations.append(
            gx.expectations.ExpectColumnValuesToBeInSet(column=column, value_set=value_set)
        )

    # === NUMERIC RANGE VALIDATION ===
    logger.info("   📊 Validating numeric ranges and business constraints...")
    for column, (min_value, max_value, mostly) in NUMERIC_RANGES.items():
        kwargs = {"column": column, "min_value": min_value, "max_value": max_value}
        if mostly is not None:
            kwargs["mostly"] = mostly
        expectations.append(gx.expectations.ExpectColumnValuesToBeBetween(**kwargs))

    # No missing values in identifying/critical fields
    for column in NOT_NULL_COLUMNS:
        expectations.append(gx.expectations.ExpectColumnValuesToNotBeNull(column=column))

    # === RUN VALIDATION SUITE ===
    logger.info("   ⚙️  Running complete validation suite...")
    results = [batch.validate(expectation) for expectation in expectations]

    # === PROCESS RESULTS ===
    failed_expectations = [
        r.expectation_config.type for r in results if not r.success
    ]

    total_checks = len(results)
    passed_checks = sum(1 for r in results if r.success)
    failed_checks = total_checks - passed_checks
    success = failed_checks == 0

    if success:
        logger.info(f"✅ Data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        logger.info(f"❌ Data validation FAILED: {failed_checks}/{total_checks} checks failed")
        logger.info(f"   Failed expectations: {failed_expectations}")

    return success, failed_expectations
