# S1 Linear Family

This folder contains the Linear Family implementation and notebooks for the
S1 workstream.

## Important data rule

For final submission, I have kept the S1 input datasets and processed split CSV files
under `data/` so the workflows can run without manual dataset setup.
Bulky generated model artifacts remain local/ignored and are recreated by the
runners.

For the Week 3 UCI Concrete baseline, the input file lives here:

```text
data/processed/uci_concrete_clean_engineered.csv
```

For Week 3, only the original Yeh/UCI input features are used:

- Cement
- Slag
- FlyAsh
- Water
- Superplasticizer
- CoarseAggregate
- FineAggregate
- Age

The engineered Week 2 columns may exist in the CSV, but they are ignored in this Week 3 baseline.

## Setup

From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run Week 3

From `S1_Linear/`:

```bash
PYTHONPATH=src python scripts/run_week03_linear.py
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH="src"
python scripts/run_week03_linear.py
```

## Generated outputs

These files are recreated locally when the script runs and are ignored by Git by default:

- Metrics table: `reports/tables/week03_linear_metrics.csv`
- Best parameters: `reports/tables/week03_linear_best_params.csv`
- Metrics copy: `results/metrics/week03_linear_metrics.csv`
- Predictions: `results/predictions/week03_linear_predictions.csv`
- Figures: `reports/figures/`
- Models: `results/models/`

## Run Week 6

Week 6 prepares the UHPC modeling dataset with semantic missingness rules.
It starts from:

```text
data/processed/uhpc_rows_with_28day_target.csv
```

From `S1_Linear/`:

```bash
PYTHONPATH=src python scripts/run_week06_semantic_missingness.py
```

The script creates:

- `data/processed/week6_semantic_cleaned.csv`
- `data/processed/week6/full_raw/X_train.csv`, `X_val.csv`, `X_test.csv`
- `data/processed/week6/raw_le_50/X_train.csv`, `X_val.csv`, `X_test.csv`
- `data/processed/week6/raw_le_20/X_train.csv`, `X_val.csv`, `X_test.csv`
- Policy-level `numeric_fully_missing_audit.csv`, `categorical_cardinality_report.csv`,
  `preprocessing_summary.csv`, and `modeling_features.csv`
- Week 6 audit tables in `reports/tables/`
- Leakage-safe train-fitted preprocessors in `results/models/week6_preprocessors/`

## Prepare the Shared UHPC Semantic Dataset

From Week 7 onward, the Linear Family work uses the shared
semantic-recoded **50 percent policy** UHPC representation. Local copies live
under:

```text
data/processed/shared_strategies/uhpc_semantic_50/
```

The active modeling files are:

- `uhpc_semantic_50_modeling.csv` for Week 7 row-mixed modeling;
- `uhpc_semantic_50_publication_ready.csv` for Week 8/9 publication-aware
  splitting and diagnostics.

Prepare the shared dataset for model training:

```bash
PYTHONPATH=src python scripts/run_week07_preprocess_shared_uhpc.py
```

The preprocessing script:

- creates feature-hash grouped 70/15/15 train, validation, and test splits;
- fits median imputation and `StandardScaler` on numeric training features
  only;
- fits `OneHotEncoder(handle_unknown='ignore')` on the fixed low-cardinality
  shared categorical group;
- fits `TargetEncoder(cv=5)` on the fixed high-cardinality shared categorical
  group and scales those encoded columns;
- produces 60 transformed model-input columns for the shared 50 percent
  semantic setup;
- saves the fitted preprocessor to
  `results/models/week7_semantic_50_preprocessor.joblib`;
- saves raw splits, transformed inspection copies, and preprocessing audits.

For model selection and cross-validation, use the raw files in
`data/processed/week7/semantic_50_splits/` and place a fresh Week 7
preprocessor inside each model pipeline. The transformed CSV files are saved
for inspection and fixed-split checks; they should not be used for
cross-validation because their preprocessor was fitted on the complete Week 7
training split.

## Run Week 7 Linear Family experiments

From `S1_Linear/`:

```bash
PYTHONPATH=src python scripts/run_week07_linear_experiments.py
```

The runner tunes model parameters from
`configs/week07_linear_experiments.yaml` using group-aware `GridSearchCV` on
training rows only. It trains OLS, Elastic Net, Bayesian Ridge, and Polynomial
Ridge. The tuned settings are frozen before running:

- curing-regime and fiber-group error analysis for the best validation model;
- fiber-feature ablation;
- training-target outlier sensitivity;
- exploratory engineering-ratio features;
- numeric-feature VIF as a separate OLS coefficient-stability diagnostic.

## Build the Week 8 publication-held-out foundation

Week 8 evaluates whether the Linear Family models generalize to publications
that were completely unseen during training. All Week 8 modules live under
`src/s1_linear/week08/`.

From `S1_Linear/`:

```bash
PYTHONPATH=src python -m s1_linear.week08.runner
```

Inside the Week 8 runner:

- derives publication lineage from the shared publication-ready dataset and
  aligns it with the exact 2,073-row Week 7 modeling input;
- verifies that publication metadata is absent from model predictors;
- creates publication-level composition, target, fiber, curing, and
  missingness audits;
- creates one shared size-balanced publication-held-out split;
- saves predictor, target, and lineage files separately for every split;
- verifies zero publication overlap between train, validation, and test;
- tunes the four Linear Family models using publication-group cross-validation;
- selects on held-out validation publications and evaluates the selected model
  once on unseen test publications;
- compares the same frozen model against the row-mixed Week 7 split;
- runs leave-one-publication-out analysis for eligible publications;
- saves worst-publication, worst-row, and presentation-figure outputs.

## Run the Week 9 uncertainty-calibration workflow

Week 9 reuses the frozen Week 8 publication split and model configuration. It
fits models on training publications only, calibrates intervals on validation
publications, and evaluates the unchanged test publications.

From `S1_Linear/`:

```bash
PYTHONPATH=src .venv/bin/python -m s1_linear.week09.runner
```

The runner evaluates 90% Elastic Net split-conformal intervals, native and
conformalized Bayesian Ridge intervals, and raw and conformalized residual
bootstrap intervals. It reports coverage, interval width, Winkler score,
publication confidence diagnostics, and calibrated LOPO results for the six
publications meeting the Week 8 50-row threshold.

Feature-policy sensitivity is intentionally skipped because Week 7 uses only
the agreed semantic-recoded 50 percent policy.

The runner also saves Week 7 interpretation figures in
`reports/figures/week07/`, including model metric comparisons, selected
baseline prediction plots, group-error plots, VIF, experiment comparisons, and
the selected baseline coefficient diagnostic.

The executed explanatory notebook is:

```text
notebooks/week07_linear_family_results.ipynb
```
