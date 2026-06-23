import great_expectations as gx
from typing import Tuple, List


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
    print("🔍 Starting data validation with Great Expectations...")

    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas("ecommerce_pandas_datasource")
    data_asset = data_source.add_dataframe_asset(name="ecommerce_data")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("batch_definition")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    expectations = []

    # === SCHEMA VALIDATION - ESSENTIAL COLUMNS ===
    print("   📋 Validating schema and required columns...")
    expectations.append(gx.expectations.ExpectColumnToExist(column="CustomerID"))
    expectations.append(gx.expectations.ExpectColumnValuesToNotBeNull(column="CustomerID"))

    # Demographics
    expectations.append(gx.expectations.ExpectColumnToExist(column="Gender"))
    expectations.append(gx.expectations.ExpectColumnToExist(column="MaritalStatus"))

    # Engagement / account features (key churn predictors)
    expectations.append(gx.expectations.ExpectColumnToExist(column="Tenure"))
    expectations.append(gx.expectations.ExpectColumnToExist(column="PreferredLoginDevice"))
    expectations.append(gx.expectations.ExpectColumnToExist(column="PreferredPaymentMode"))
    expectations.append(gx.expectations.ExpectColumnToExist(column="SatisfactionScore"))
    expectations.append(gx.expectations.ExpectColumnToExist(column="Complain"))

    # === BUSINESS LOGIC VALIDATION ===
    print("   💼 Validating business logic constraints...")
    expectations.append(gx.expectations.ExpectColumnValuesToBeInSet(column="Gender", value_set=["Male", "Female"]))
    expectations.append(gx.expectations.ExpectColumnValuesToBeInSet(
        column="MaritalStatus", value_set=["Single", "Married", "Divorced"]
    ))
    expectations.append(gx.expectations.ExpectColumnValuesToBeInSet(
        column="PreferredLoginDevice", value_set=["Mobile Phone", "Phone", "Computer"]
    ))
    expectations.append(gx.expectations.ExpectColumnValuesToBeInSet(column="CityTier", value_set=[1, 2, 3]))
    expectations.append(gx.expectations.ExpectColumnValuesToBeInSet(column="Complain", value_set=[0, 1]))

    # === NUMERIC RANGE VALIDATION ===
    print("   📊 Validating numeric ranges and business constraints...")

    # Tenure (months with the platform) can't be negative
    expectations.append(gx.expectations.ExpectColumnValuesToBeBetween(
        column="Tenure", min_value=0, max_value=120, mostly=0.99
    ))

    # Satisfaction is a 1-5 rating scale
    expectations.append(gx.expectations.ExpectColumnValuesToBeBetween(
        column="SatisfactionScore", min_value=1, max_value=5
    ))

    # Distance from warehouse to home must be positive
    expectations.append(gx.expectations.ExpectColumnValuesToBeBetween(
        column="WarehouseToHome", min_value=0, mostly=0.99
    ))

    # Cashback amount can't be negative
    expectations.append(gx.expectations.ExpectColumnValuesToBeBetween(column="CashbackAmount", min_value=0))

    # No missing values in identifying/critical fields
    expectations.append(gx.expectations.ExpectColumnValuesToNotBeNull(column="Gender"))
    expectations.append(gx.expectations.ExpectColumnValuesToNotBeNull(column="SatisfactionScore"))

    # === RUN VALIDATION SUITE ===
    print("   ⚙️  Running complete validation suite...")
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
        print(f"✅ Data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        print(f"❌ Data validation FAILED: {failed_checks}/{total_checks} checks failed")
        print(f"   Failed expectations: {failed_expectations}")

    return success, failed_expectations
