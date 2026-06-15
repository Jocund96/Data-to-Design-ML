# S1 Linear Family — Week 3 Baselines

This folder contains the Linear Family implementation for the Week 3 UCI Concrete baseline task.

## Important data rule

Datasets are **not committed** to GitHub. Place your local CSV here:

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

## Import the teammate-prepared UHPC dataset

The current-week Linear Family work uses the teammate's semantic-recoded
**50 percent policy** representation. A correction step restores the valid row
accidentally skipped upstream, and the import step creates the S1-owned
modeling input.

From `S1_Linear/`:

```bash
PYTHONPATH=src python scripts/run_week07_import_teammate_uhpc.py
```

The import script:

- restores the one valid mix accidentally skipped upstream with
  `scripts/run_week07_correct_semantic_dataset.py`;
- imports the corrected S1-owned
  `data/processed/week7/semantic_recoding_features_50_corrected_2073.csv`;
- drops the accidental `Unnamed: 0` saved-index predictor;
- keeps `cement_type_clean` and drops the redundant `cement_type`;
- removes exact feature-and-target duplicate rows;
- saves `data/processed/week7/uhpc_teammate_semantic_50_linear_ready.csv`;
- saves lineage, column, and readiness audits in `reports/tables/`.

Then prepare the agreed dataset for model training:

```bash
PYTHONPATH=src python scripts/run_week07_preprocess_teammate_uhpc.py
```

The preprocessing script:

- creates feature-hash grouped 70/15/15 train, validation, and test splits;
- fits median imputation, missing indicators, and `StandardScaler` on numeric
  training features only;
- fits rare-aware one-hot encoding on categorical training features only;
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
