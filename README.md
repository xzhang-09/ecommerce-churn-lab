# E-Commerce Customer Churn Pipeline

*From raw data to a calibrated churn model, a cost-aware operating threshold, and a randomized retention experiment — reproducible and leakage-free.*

## Purpose

The data is a public e-commerce customer dataset — ~5,000 customers, ~17% churn after deduplication, with `Churn` as the target label.

The project is intentionally scoped to **offline analysis and modeling** — EDA, data-quality checks, feature engineering, training, experiment tracking, evaluation, batch scoring, and randomized-experiment design for the retention campaign (assignment generation only; no outcome data or causal estimates yet).

There is no real-time serving layer (Docker, FastAPI, Gradio): with only a single static public snapshot and no live feature source or deployment target, an API would be mostly a demo artifact rather than a meaningful production path. Churn scoring is represented as scheduled batch scoring via `churn-score`. Production monitoring, drift tracking, and model-registry promotion across retrains would require live traffic and data arriving over time, so this project focuses on what the static dataset can support reliably: leakage-free modeling, calibration, reproducible tracking, and offline decisioning.

## Highlights

- Validated offline churn pipeline: Great Expectations data checks, leakage-free preprocessing, model training, MLflow tracking, and batch scoring
- Latest validated run: test PR-AUC `0.931`, with recall `0.982` / precision `0.671` under a high-recall retention operating threshold
- Calibrated probabilities (`--calibrate isotonic` by default) with Brier/ECE reported for expected-value targeting
- Cross-validated RandomForest / LightGBM / XGBoost comparison (with optional Optuna tuning), plus permutation feature importance on the held-out test split
- Retention decisioning beyond prediction: cost-aware thresholding and a pre-specified coupon experiment assignment/analysis workflow

## Project Structure

```text
.
├── .github/workflows/       # CI: compileall + pytest on push / pull request
├── data/
│   ├── raw/                 # Raw dataset, ignored by git
│   └── processed/           # Processed outputs, ignored by git
├── docs/
│   ├── COUPON_EXPERIMENT_DESIGN.md  # Coupon experiment design + pre-analysis plan
│   └── RESULTS.md           # Generated latest local metrics report
├── notebooks/
│   └── EDA.ipynb            # Exploratory analysis
├── scripts/                 # Thin CLI wrappers around the churn package
│   ├── analyze_coupon_experiment.py
│   ├── assign_coupon_experiment.py
│   ├── prepare_processed_data.py
│   ├── run_pipeline.py
│   └── score_customers.py
├── src/
│   └── churn/               # Installable package (src-layout)
│       ├── data/            # load_data, preprocess, prepare
│       ├── experiments/     # coupon_assignment + coupon_analysis (randomized experiment)
│       ├── features/        # build_features (feature schema fit/apply)
│       ├── models/          # estimators, evaluate, training, compare, tune, score
│       ├── validation/      # validate_data (Great Expectations)
│       ├── logging_utils.py # package logging setup
│       └── pipeline.py      # orchestration (stages + CLI)
├── tests/
│   ├── unit/                # fast unit tests
│   ├── integration/         # smoke test + full pipeline.main run
│   ├── test_project_structure.py  # package layout / import checks
│   └── test_notebook_scope.py     # notebook scope guardrail
├── pyproject.toml
└── README.md
```

## Setup

Python 3.11+ is required; 3.11 is the tested/recommended version.

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies and the `churn` package (editable, src-layout). Choose one:

```bash
# Runtime only:
python -m pip install -e .

# Development / tests / notebooks:
python -m pip install -e ".[dev]"
```

The editable install makes `import churn...` and the `churn-pipeline` / `churn-prepare` /
`churn-score` / `churn-assign-coupon-experiment` / `churn-analyze-coupon-experiment`
console scripts work from anywhere — no `sys.path` manipulation required.
Runtime dependencies live in `[project]`; test and notebook tooling is in the `dev`
optional-dependencies group so a plain install stays lean.

## Data

Place the raw dataset at:

```text
data/raw/E Commerce Dataset.xlsx
```

The raw dataset is not redistributed in this repository; download a licensed
copy from Kaggle (source below) and place it at that path before running the
pipeline. The loader uses the `E Comm` sheet when it exists; otherwise it falls
back to the first Excel sheet.

**Data source:** [*Ecommerce Customer Churn Analysis and Prediction*](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction)
by Ankit Verma on Kaggle — 5,630 customers × 20 columns, with the schema
documented in the workbook's `Data Dict` sheet and the records in `E Comm`.

The test suite and CI use synthetic data, so reviewers can run compile/tests
without access to the raw Excel file.

Dataset snapshot from the latest generated run (`docs/RESULTS.md`):

- Raw rows: 5,630
- Rows after exact duplicate removal: 5,073
- Churn rate after cleaning: 16.6% (`841 / 5,073`)
- Target column: `Churn`

## Run The Pipeline

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

For a quick local modeling run without the validation gate:

```bash
python scripts/run_pipeline.py \
  --input "data/raw/E Commerce Dataset.xlsx" \
  --target Churn \
  --skip_validation
```

Optional Optuna tuning for LightGBM or XGBoost:

```bash
python scripts/run_pipeline.py \
  --input "data/raw/E Commerce Dataset.xlsx" \
  --target Churn \
  --model lightgbm \
  --tune \
  --tune_metric average_precision \
  --tune_trials 30
```

## Latest Local Results

[docs/RESULTS.md](docs/RESULTS.md) is generated by the pipeline and is the
source of truth for the latest local run: run provenance, held-out test metrics,
the cross-validated model comparison, and permutation-importance drivers. A
successful pipeline run regenerates it. The headline numbers below are copied by
hand from that report, so re-running the `--compare_models` command above
refreshes both.

Headline held-out test results for the final calibrated LightGBM model:

| Metric | Value |
|---|---:|
| Test PR-AUC | 0.931 |
| Test ROC-AUC | 0.984 |
| Recall @ threshold 0.114 | 0.982 (catches 165 of 168 churners) |
| Precision @ threshold 0.114 | 0.671 (81 false alarms) |
| Business cost (`10 * FN + FP`) | 111 FP-equivalents |
| Calibration Brier / ECE | 0.0317 / 0.0191 |

The very high recall is by design: the low 0.114 threshold is selected by the
10:1 missed-churner cost objective, which favors high recall (see
[Business Interpretation](#business-interpretation)). The high ROC-AUC should be
read in context: this static dataset appears strongly separable, with features
such as `Tenure` and `Complain` carrying much of the signal. The result reflects
dataset separability plus leakage-control hygiene (pre-split dedup, three-way
split, out-of-fold threshold selection), not evidence of production performance.
LightGBM wins on CV PR-AUC, but the candidates are within ~1 CV std; see
[docs/RESULTS.md](docs/RESULTS.md) for the full comparison and top drivers.

These remain offline results on a static dataset — a production system would still
need fresh holdout data, monitoring, and a tested inference path.

## Interpreting Results

The model comparison is an empirical ranking under the current data split,
features, and comparison-time default model settings — "slightly ahead in this
run", not a definitive best model. In the comparison table, PR-AUC drives the
model ranking; precision, recall, F1, and cost are out-of-fold operating-point
context at each model's selected threshold.

PR-AUC is the primary ranking metric because churn is imbalanced (~17%
positives): it is sensitive to precision on the minority class, whereas ROC AUC
can look strong even when precision is weak.

## Business Interpretation

The default threshold objective treats a missed churner as 10 times more
expensive than a false alarm:

```text
business_cost = 10 * false_negatives + false_positives
```

This is a simple decision model for retention targeting. A lower threshold
catches more likely churners but contacts more customers who would not have
churned. A higher threshold reduces unnecessary outreach but misses more
churners. Change `--fn_fp_ratio` when the retention offer cost, customer lifetime
value, or intervention success rate differs from the default assumption.

For a real retention program, the next step is to replace this simple FN/FP ratio
with expected value:

```text
expected_value = churn_probability * expected_margin_saved * intervention_success_rate - offer_cost
```

That would turn the model from "who is likely to churn" into "who is worth
contacting." Because the pipeline calibrates `churn_probability` (default
isotonic, checked with the reported Brier/ECE), the probability can be used as
one input to this expected-value formula rather than treated as a bare ranking
score. The remaining inputs — expected margin saved, intervention success rate,
and offer cost — must come from business data or experiment results.

## Reusing on Another Dataset

The trained model is specific to this data; the pipeline is not. Adapting it to
another churn dataset means swapping the data-specific pieces and re-running the
same stages:

- **You supply:** the labeled dataset and its churn definition, the available
  input fields, the Great Expectations rules, the category-alias map, and the
  FN:FP cost ratio.
- **Reused unchanged:** the split / imputation / calibration hygiene, the
  cross-validated model comparison, threshold selection, MLflow's matched
  *(model + preprocessing + threshold)* bundle, batch scoring, and the
  randomized-experiment workflow.

## Limitations

- The evaluation uses a random held-out split, not a time-based split. A future
  production version should validate on later customer cohorts to measure drift.
- The headline test metrics are point estimates from a single random split of a
  static dataset. No confidence intervals are reported for them, so treat them as
  indicative of this split rather than guaranteed, and note that they do not
  replace external validation on fresh data.
- `Tenure` is the strongest predictor, which is plausible, but it should also be
  checked against how churn labels were constructed before drawing product
  conclusions.
- Permutation importance describes predictive reliance, not causal effect: this
  static observational dataset can show which signals the model uses, but not
  whether changing those factors would reduce churn.
- The coupon workflow creates assignments and a pre-specified analysis path only;
  without post-campaign outcomes for treatment and control customers, it cannot
  estimate coupon lift, causal impact, or incremental profit yet.

## Generated Outputs

`run_pipeline.py` (`churn-pipeline`) writes:

- Cleaned data to `data/processed/ecommerce_churn_cleaned.csv`
- Feature/encoding metadata to `artifacts/` (`feature_columns.json`, `feature_schema.json`, and `preprocessing.pkl` — the complete leakage-free preprocessing state: feature columns, categorical schema, and the training-fit imputation medians)
- Permutation feature importance to `artifacts/feature_importance.csv`
- Model comparison results to `artifacts/model_comparison.csv` (only with `--compare_models`)
- A regenerated Markdown results report to `docs/RESULTS.md` (run provenance, held-out test metrics, model comparison, top importance drivers)
- MLflow runs to `mlruns/`, each bundling:
  - run parameters and the trained calibrated model artifact
  - `preprocessing.pkl`, `threshold.txt`, and the training-fit imputation medians (`imputation_medians.json`)
  - metrics: precision, recall, F1, ROC AUC, PR AUC, the FN/FP business cost, calibration diagnostics (Brier / ECE on test and out-of-fold), confusion-matrix counts, and 5-fold CV stability

`prepare_processed_data.py` (`churn-prepare`) is a separate data-prep step that writes both the cleaned table and the model-ready feature matrix:

- `data/processed/ecommerce_churn_cleaned.csv`
- `data/processed/ecommerce_churn_features.csv`

View local MLflow runs after at least one successful pipeline run:

```bash
mlflow ui --backend-store-uri file:./mlruns
```

## Batch Scoring

After at least one successful pipeline run has created an MLflow run, score a
fresh batch of customers into a ranked contact list:

```bash
python scripts/score_customers.py \
  --input "data/raw/new_customers.xlsx" \
  --output data/processed/churn_scores.csv
```

`score_customers.py` (`churn-score`) loads the model, the leakage-free preprocessing
state (`preprocessing.pkl`), and the selected decision threshold from a **single MLflow
run** — the latest by default, or `--run-id <id>` for a specific one. Loading all three
from the same run guarantees they are a matched set (no model-from-one-run /
threshold-from-another mismatch). The new batch is transformed exactly as in training
(category aliases, missing-value indicators, the fitted feature schema, and the
*training* imputation medians), then written one row per customer as
`(customer id when available, churn_probability, churn_prediction)`, sorted
highest-risk first.

## Beyond Prediction: Coupon Experiment

The scored list answers "who is likely to churn," not "who is worth sending a
coupon." To test whether an intervention changes outcomes, turn the scored file
into a blocked randomized assignment:

```bash
python scripts/assign_coupon_experiment.py \
  --input data/processed/churn_scores.csv \
  --output data/processed/coupon_experiment_assignments.csv \
  --risk-pool-fraction 0.30 --treatment-fraction 0.50 --risk-bands 3 --random-state 42
```

`assign_coupon_experiment.py` (`churn-assign-coupon-experiment`) keeps the top-risk
customers, splits them into ordered risk bands, randomizes treatment/control within
each band, validates the input, and writes an audit `*_metadata.json` next to the CSV.
The scored file must include a customer id column, `CustomerID` by default.

The full design — population choice, outcome definitions, intent-to-treat analysis
plan, sample-size considerations, and limitations — is in
`docs/COUPON_EXPERIMENT_DESIGN.md`. The project has no post-assignment outcome data
yet, so this produces the experiment inputs, not causal estimates.

Once the campaign window closes and a binary outcome has been collected for every
assigned customer, run the pre-specified intent-to-treat analysis:

```bash
python scripts/analyze_coupon_experiment.py \
  --assignments data/processed/coupon_experiment_assignments.csv \
  --outcomes data/processed/coupon_experiment_outcomes.csv \
  --output data/processed/coupon_experiment_analysis.json
```

`analyze_coupon_experiment.py` (`churn-analyze-coupon-experiment`) locks the analysis
plan before outcomes exist: overall ITT effect, Wald confidence interval, within-band
randomization-test p-value, per-band breakdown, and JSON input fingerprints. It fails
if any assigned customer is missing an outcome row, because silently dropping rows
would undo the randomization.

## Testing

```bash
python -m compileall src scripts
python -m pytest -q
```

Tests live under `tests/unit/` (fast unit tests) and `tests/integration/` (a
lightweight smoke test plus a full `pipeline.main` run, both on synthetic data —
no Excel required). The same checks run in CI on every push and pull request via
`.github/workflows/ci.yml`.

## Notes

- `data/raw/`, `data/processed/`, `artifacts/`, and `mlruns/` are generated locally and are git-ignored.
- Outputs resolve relative to the current working directory, so run the pipeline from the repo root.
