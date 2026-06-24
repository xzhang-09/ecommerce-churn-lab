# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Scope

This repository is scoped as an offline e-commerce customer churn analysis and modeling project. It does not include Docker, cloud deployment, FastAPI, Gradio, or model serving components.

## Commands

### Environment

```bash
# Recommended Python version: 3.11
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# Install the `churn` package itself (editable, src-layout). This makes
# `import churn...` and the console scripts work without any sys.path hacks.
python -m pip install -e .
```

The code lives in an installable `churn` package under `src/` (src-layout). After
`pip install -e .` two console scripts are available: `churn-pipeline` (the modeling
pipeline) and `churn-prepare` (processed-data prep). The `scripts/*.py` files are
thin wrappers kept for the documented `python scripts/...` invocations.

### Analysis Pipeline

```bash
# Run the complete data validation, preprocessing, feature engineering, training, and evaluation pipeline.
python scripts/run_pipeline.py --input "data/raw/E Commerce Dataset.xlsx" --target Churn

# Quick local run when Great Expectations is unavailable.
python scripts/run_pipeline.py --input "data/raw/E Commerce Dataset.xlsx" --target Churn --skip_validation

# Fairly compare RandomForest, LightGBM, and XGBoost (5-fold CV PR-AUC), then train a selected final model.
python scripts/run_pipeline.py --input "data/raw/E Commerce Dataset.xlsx" --target Churn --compare_models --model lightgbm

# Tune the final boosting model with Optuna (xgboost or lightgbm; applies to the final model only, not the comparison).
python scripts/run_pipeline.py --input "data/raw/E Commerce Dataset.xlsx" --target Churn --model lightgbm --tune --tune_metric average_precision --tune_trials 30

# Prepare processed data only.
python scripts/prepare_processed_data.py        # or: churn-prepare
```

The pipeline can also be invoked as a console script after `pip install -e .`,
e.g. `churn-pipeline --input "data/raw/E Commerce Dataset.xlsx" --target Churn`.

### Verification

```bash
python -m compileall src scripts
python -m pytest -q
```

Tests live under `tests/` (`tests/unit/` for fast unit tests, `tests/integration/`
for the end-to-end smoke test). The former `scripts/test_pipeline_phase{1,2}_*.py`
manual-check scripts were removed — their coverage is now in the pytest suites and
the reproducible pipeline itself.

## Architecture Overview

### Analysis Flow

`scripts/run_pipeline.py` orchestrates the main workflow:

1. Load the raw Excel or CSV dataset.
2. Validate schema and business constraints with Great Expectations.
3. Preprocess data by trimming headers, mapping target labels, normalizing category aliases, adding `*_missing` indicator columns, and dropping exact duplicates. Numeric NaNs are left in place here and median-imputed later (train-only) to avoid leakage.
4. Build model-ready features with deterministic binary encoding and one-hot encoding.
5. Split the data into stratified train / validation / test sets.
6. Median-impute numeric columns using medians fit on the training split only.
7. Train the final model (default LightGBM — it has the lowest FN/FP business cost and ties for the best CV PR-AUC) with early stopping on the validation split, then apply post-hoc probability calibration fit on that same validation split (`--calibrate`, default `isotonic`; `sigmoid`/Platt is steadier on very small data; `none` keeps the raw de-calibrated scores). Calibration is monotonic, so it does not change ranking or the selected threshold but makes the saved/evaluated probabilities trustworthy for expected-value targeting. Optuna tuning is available for both boosting models (`--tune` with `--model xgboost` or `--model lightgbm`; each has its own regularization-focused search space; default objective `average_precision` / PR-AUC, a threshold-free metric). Candidate estimators use plain conventional defaults; for the boosting models `n_estimators` is only an upper bound — early stopping picks the tree count.
8. Optionally compare RandomForest, LightGBM, and XGBoost with `--compare_models`. The comparison is fair and cross-validated: all candidates run at their default configs (no per-model Optuna inside the ranking — that would compare a tuned model against untuned rivals), are scored with 5-fold CV PR-AUC/ROC-AUC (early stopping inside each fold), and are ranked by CV PR-AUC mean (std reported so close ranks read as ties). Per-model precision/recall/f1 and `oof_cost` are shown for context only, computed on the out-of-fold predictions (never the held-out test set, which is reserved for the single final model).
9. Select the final decision threshold on the out-of-fold CV predictions unless an explicit `--threshold` is given. The default `--threshold_metric cost` minimizes the FN/FP business cost where a missed churner costs `--fn_fp_ratio` times a false alarm (default 10); `f1`/`precision`/`recall` are also available.
10. Evaluate once on the untouched test split, compute model-agnostic permutation importance (PR-AUC drop) on it, and log parameters, metrics (including the business cost), feature metadata, permutation importance, and model artifacts with MLflow.

Deduplication before the split is important. The raw e-commerce dataset contains repeated rows after `CustomerID` is dropped (all CustomerIDs are unique, but ~10% of rows are exact feature duplicates with consistent labels — i.e. genuine duplicate records); leaving them in place can leak duplicate rows across the split and inflate evaluation metrics.

The three-way split and validation-based threshold selection matter for evaluation hygiene: the decision threshold (and per-model choice) must be tuned on validation data, never on the test set, or the reported metrics are optimistically biased.

### Key Files

- `notebooks/EDA.ipynb`: exploratory data analysis.
- `src/churn/data/load_data.py`: loads CSV or Excel data, using the `E Comm` sheet when present.
- `src/churn/data/preprocess.py`: handles basic cleaning, category aliases, missing-value indicators, duplicate rows, and leakage-free median imputation (`impute_numeric`).
- `src/churn/data/prepare.py`: creates processed data without running model training.
- `src/churn/features/build_features.py`: creates model-ready features.
- `src/churn/validation/validate_data.py`: validates the raw dataset with Great Expectations (rules are module-level constants at the top of the file).
- `src/churn/models/tune.py`: runs Optuna hyperparameter tuning for XGBoost or LightGBM (model-specific search spaces; `build_tuned_estimator` rebuilds the tuned model).
- `src/churn/pipeline.py`: primary reproducible modeling pipeline (library form; `build_arg_parser`/`cli`/`main`).
- `scripts/run_pipeline.py`, `scripts/prepare_processed_data.py`: thin CLI wrappers around the package.

### MLflow

- Default experiment: `E-Commerce Churn`
- Default tracking URI: `file://{project_root}/mlruns`
- Logged artifacts: `model/` (the calibrated estimator actually used for inference, unless `--calibrate none`), `feature_columns.txt`, `threshold.txt` (the selected threshold), `preprocessing.pkl`, `calibration_reliability_test.csv`, `feature_importance.csv` (permutation importance), and `model_comparison.csv` when `--compare_models` is used
- Logged metrics: `precision`, `recall`, `f1`, `roc_auc`, `pr_auc`, `test_cost` (FN/FP business cost), `brier`/`ece` (calibration, test) and `brier_oof`/`ece_oof` (calibration, out-of-fold), confusion counts (`tp`/`fp`/`fn`/`tn`), `cv_pr_auc_mean`/`_std`, `cv_roc_auc_mean`/`_std`, `best_iteration`, `train_time`, `pred_time`, `data_quality_pass`. Nested `--compare_models` runs log `oof_cost` per candidate.

View local runs with:

```bash
mlflow ui --backend-store-uri file:./mlruns
```

### Generated Data Files

- `data/processed/ecommerce_churn_cleaned.csv`: cleaned but not feature-encoded data.
- `data/processed/ecommerce_churn_features.csv`: model-ready feature matrix, including the target column.
- `artifacts/model_comparison.csv`: cross-validated model comparison (CV PR-AUC/ROC-AUC mean ± std plus out-of-fold precision/recall/f1/`oof_cost`) when `--compare_models` is used.
- `artifacts/feature_importance.csv`: model-agnostic permutation importance (mean/std PR-AUC drop on the test split), ranked descending.

## Repository Hygiene

- `data/`, `artifacts/`, and `mlruns/` are generated local artifacts and should not be committed unless explicitly required.
- Keep changes focused on analysis, modeling, validation, and documentation.
- Do not reintroduce deployment infrastructure unless the project scope changes again.
