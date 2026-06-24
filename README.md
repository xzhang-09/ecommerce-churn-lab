# E-Commerce Customer Churn Analysis

### Purpose

Analyze customer churn in an e-commerce dataset (~5,000 customers, ~17% churn after dedup) and build a reproducible, leakage-free pipeline that produces a churn model, a calibrated risk score, and a business operating threshold.

The project is intentionally scoped to **offline analysis and modeling** — EDA, data quality checks, feature engineering, training, experiment tracking, and evaluation. It is not an online service: Docker, FastAPI, Gradio, and serving artifacts have been deliberately removed to keep the boundary clear. Productionizing it would still require a serving interface, CI/CD, monitoring, model-registry promotion rules, and a tested batch/real-time inference path.

### What This Project Covers

- Exploratory data analysis in `notebooks/EDA.ipynb`
- Raw data loading from Excel or CSV files
- Data validation with Great Expectations
- Preprocessing for missing values (missingness indicators + leakage-free median imputation), duplicate records, and inconsistent category labels
- Feature engineering for categorical and numeric fields
- Fair, cross-validated RandomForest / LightGBM / XGBoost comparison (all at default configs, ranked by 5-fold CV PR-AUC with early stopping)
- Early stopping for the boosting models, so the tree count is chosen from data rather than hardcoded
- Optional Optuna tuning for the final boosting model (XGBoost or LightGBM, each with its own regularization-focused search space; default objective: PR-AUC)
- Leakage-free evaluation: stratified train/validation/test split, with the decision threshold selected on out-of-fold CV predictions and the test set scored only once
- Cost-aware threshold selection: by default the threshold minimizes the FN/FP business cost (a missed churner costs `--fn_fp_ratio`x a false alarm, default 10:1)
- Post-hoc probability calibration (`--calibrate`, default isotonic) so the final probabilities are trustworthy for expected-value targeting, with Brier/ECE reported on test and out-of-fold
- Model-agnostic permutation feature importance (PR-AUC drop) saved to `artifacts/feature_importance.csv`
- MLflow experiment tracking for parameters, metrics, and model artifacts

### Project Structure

```text
.
├── data/
│   ├── raw/                 # Raw dataset, ignored by git
│   └── processed/           # Processed outputs, ignored by git
├── notebooks/
│   └── EDA.ipynb            # Exploratory analysis
├── scripts/                 # Thin CLI wrappers around the churn package
│   ├── prepare_processed_data.py
│   └── run_pipeline.py
├── src/
│   └── churn/               # Installable package (src-layout)
│       ├── data/            # load_data, preprocess, prepare
│       ├── features/        # build_features
│       ├── models/          # tune
│       ├── validation/      # validate_data (Great Expectations)
│       └── pipeline.py      # end-to-end modeling pipeline
├── tests/
│   ├── unit/                # fast unit tests
│   └── integration/         # end-to-end smoke test
├── pyproject.toml
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

Install dependencies and the `churn` package (editable, src-layout):

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

The editable install makes `import churn...` and the `churn-pipeline` / `churn-prepare`
console scripts work from anywhere — no `sys.path` manipulation required.

### Data

Place the raw dataset at:

```text
data/raw/E Commerce Dataset.xlsx
```

The loader uses the `E Comm` sheet when it exists; otherwise it falls back to the first Excel sheet.

### Run The Analysis Pipeline

Run the full training and evaluation pipeline (after `pip install -e .` the
`churn-pipeline` console script is an exact equivalent of `python scripts/run_pipeline.py`):

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

`run_pipeline.py` (`churn-pipeline`) writes:

- Cleaned data to `data/processed/ecommerce_churn_cleaned.csv`
- Feature/encoding metadata to `artifacts/` (`feature_columns.json`, `feature_schema.json`, `preprocessing.pkl`)
- Permutation feature importance to `artifacts/feature_importance.csv`
- Model comparison results to `artifacts/model_comparison.csv` (only with `--compare_models`)
- MLflow runs to `mlruns/`: parameters, the trained (calibrated) model, the selected threshold, and metrics — precision, recall, F1, ROC AUC, PR AUC, the FN/FP business cost, calibration diagnostics (Brier, ECE on test and out-of-fold), confusion-matrix counts, and 5-fold CV stability

`prepare_processed_data.py` (`churn-prepare`) is a separate data-prep step that writes both the cleaned table and the model-ready feature matrix:

- `data/processed/ecommerce_churn_cleaned.csv`
- `data/processed/ecommerce_churn_features.csv`

View local MLflow runs:

```bash
mlflow ui --backend-store-uri file:./mlruns
```

### Testing

```bash
python -m compileall src scripts
python -m pytest -q
```

Tests live under `tests/unit/` (fast unit tests) and `tests/integration/` (an
end-to-end smoke test on synthetic data, no Excel required).

### Notes

- `data/`, `artifacts/`, and `mlruns/` are generated locally and are git-ignored.
- Outputs resolve relative to the current working directory, so run the pipeline from the repo root.

### Reading The Results

The model comparison is an empirical ranking under the current data split, features, and default model settings — read the latest numbers from `artifacts/model_comparison.csv` rather than a hardcoded winner. The candidates typically land within roughly one CV standard deviation of each other, so close ranks should be read as ties ("slightly ahead in this run"), not a definitive best model.

PR-AUC is the primary ranking metric because churn is imbalanced (~17% positives): it is sensitive to precision on the minority class, whereas ROC AUC can look strong even when precision is weak.

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

That would turn the model from "who is likely to churn" into "who is worth contacting." Because the pipeline now calibrates `churn_probability` (default isotonic, verified by the reported Brier/ECE), it can be plugged into this expected-value formula directly rather than being treated as a bare ranking score.
