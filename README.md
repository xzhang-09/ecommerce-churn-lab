## E-Commerce Customer Churn Analysis

### Purpose

Analyze customer churn patterns in an e-commerce dataset and build a reproducible machine-learning pipeline for churn prediction.

This project is intentionally scoped as a data analysis and modeling project. It focuses on exploratory analysis, data quality checks, feature engineering, model training, experiment tracking, and evaluation.

### Scope Decision

This repository is now optimized for offline analysis rather than online serving. The previous Docker, FastAPI, Gradio, and checked-in serving artifacts have been removed so the project boundary is clearer: produce a reproducible churn model evaluation and business operating threshold, not a production API.

Production deployment would still require a serving interface, CI/CD, monitoring, model registry promotion rules, and a tested batch or real-time inference path.

### What This Project Covers

- Exploratory data analysis in `notebooks/EDA.ipynb`
- Raw data loading from Excel or CSV files
- Data validation with Great Expectations
- Preprocessing for missing values (missingness indicators + leakage-free median imputation), duplicate records, and inconsistent category labels
- Feature engineering for categorical and numeric fields
- Fair, cross-validated RandomForest / LightGBM / XGBoost comparison (all at default configs, ranked by 5-fold CV PR-AUC with early stopping)
- Early stopping for the boosting models, so the tree count is chosen from data rather than hardcoded
- Optional Optuna tuning for the final boosting model (XGBoost or LightGBM, each with its own regularization-focused search space; default objective: PR-AUC)
- Leakage-free evaluation: stratified train/validation/test split, with the decision threshold selected on validation and the test set scored only once
- Cost-aware threshold selection: by default the threshold minimizes the FN/FP business cost (a missed churner costs `--fn_fp_ratio`x a false alarm, default 10:1)
- MLflow experiment tracking for parameters, metrics, and model artifacts

### Project Structure

```text
.
├── data/
│   ├── raw/                 # Raw dataset, ignored by git
│   └── processed/           # Processed outputs, ignored by git
├── notebooks/
│   └── EDA.ipynb            # Exploratory analysis
├── scripts/
│   ├── prepare_processed_data.py
│   ├── run_pipeline.py
│   ├── test_pipeline_phase1_data_features.py
│   └── test_pipeline_phase2_modeling.py
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── utils/
├── requirements.txt
└── README.md
```

### Setup

Recommended Python version: 3.11.

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

### Data

Place the raw dataset at:

```text
data/raw/E Commerce Dataset.xlsx
```

The loader uses the `E Comm` sheet when it exists; otherwise it falls back to the first Excel sheet.

### Run The Analysis Pipeline

Run the full training and evaluation pipeline:

```bash
python scripts/run_pipeline.py \
  --input "data/raw/E Commerce Dataset.xlsx" \
  --target Churn
```

Run a fair, cross-validated model comparison (5-fold CV PR-AUC, early stopping per fold) and then train a selected final model:

```bash
python scripts/run_pipeline.py \
  --input "data/raw/E Commerce Dataset.xlsx" \
  --target Churn \
  --compare_models \
  --model lightgbm
```

If Great Expectations is not installed and you only want a quick local modeling run:

```bash
python scripts/run_pipeline.py \
  --input "data/raw/E Commerce Dataset.xlsx" \
  --target Churn \
  --skip_validation
```

Run with Optuna tuning (works for `--model xgboost` or `--model lightgbm`):

```bash
python scripts/run_pipeline.py \
  --input "data/raw/E Commerce Dataset.xlsx" \
  --target Churn \
  --model lightgbm \
  --tune \
  --tune_metric average_precision \
  --tune_trials 30
```

### Outputs

The pipeline writes:

- Cleaned data to `data/processed/ecommerce_churn_cleaned.csv`
- Model-ready feature data to `data/processed/ecommerce_churn_features.csv`
- Feature metadata to `artifacts/`
- Reusable categorical encoding metadata to `artifacts/feature_schema.json`
- Model comparison results to `artifacts/model_comparison.csv` when `--compare_models` is used
- MLflow runs to `mlruns/`
- Evaluation metrics including precision, recall, F1, ROC AUC, PR AUC, the FN/FP business cost, and probability-calibration diagnostics (Brier score, ECE) — plus confusion-matrix counts and 5-fold CV stability

View local MLflow runs:

```bash
mlflow ui --backend-store-uri file:./mlruns
```

### Notes

- `data/`, `artifacts/`, and `mlruns/` are treated as generated local artifacts.
- The project no longer includes Docker, cloud deployment, FastAPI, or Gradio components.
- The model is evaluated as an offline analysis artifact, not as a production service.

### Reading The Results

The model comparison should be read as an empirical ranking under the current data split, features, and default model settings. In the current saved comparison, LightGBM has the highest CV PR-AUC, but the top three models are close enough that the result should be described as "LightGBM is slightly ahead in this run" rather than "LightGBM is definitively best."

PR-AUC is the primary ranking metric because churn is imbalanced: in the processed feature table, churners are a minority class. ROC AUC is still reported, but it can look strong even when precision is weak.

### Business Interpretation

The default threshold objective treats a missed churner as 10 times more expensive than a false alarm:

```text
business_cost = 10 * false_negatives + false_positives
```

This is a simple decision model for retention targeting. A lower threshold catches more likely churners but contacts more customers who would not have churned. A higher threshold reduces unnecessary outreach but misses more churners. Change `--fn_fp_ratio` when the retention offer cost, customer lifetime value, or intervention success rate differs from the default assumption.

For a real retention program, the next step is to replace this simple FN/FP ratio with expected value:

```text
expected_value = churn_probability * expected_margin_saved * intervention_success_rate - offer_cost
```

That would turn the model from "who is likely to churn" into "who is worth contacting."
