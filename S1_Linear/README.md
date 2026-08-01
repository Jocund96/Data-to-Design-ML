# S1 Linear Family

This directory contains the Linear Family experiments for concrete
compressive-strength prediction. The work covers the UCI Concrete dataset and
the literature-derived UHPC dataset.

## Requirements

- Python 3.11 or newer
- About 2 GB of free disk space for the environment and generated outputs
- A Jupyter-compatible browser for interactive notebook use

The submitted environment uses Python 3.14.0. Package versions are fixed in
`pyproject.toml`.

## Setup

Run these commands from `S1_Linear/`.

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Please confirm the installation:

```bash
python -c "import s1_linear, sklearn, shap; print('Environment ready')"
```

## Data

The required input data is included in `data/`.

| Workflow                         | Input                                                |
| -------------------------------- | ---------------------------------------------------- |
| UCI baseline and representations | `data/processed/uci_concrete_clean_engineered.csv`   |
| UHPC import                      | `data/raw/UHPC Dataset  (Version-2).xlsx`            |
| UHPC semantic missingness        | `data/processed/uhpc_rows_with_28day_target.csv`     |
| Shared UHPC experiments          | `data/processed/shared_strategies/uhpc_semantic_50/` |

The UCI baseline uses the original eight Yeh predictors: cement, slag, fly
ash, water, superplasticizer, coarse aggregate, fine aggregate, and age.

## Complete Run

The complete workflow runs the experiment code first and then executes each
notebook with the generated outputs:

```bash
python scripts/run_all.py
```

The workflows are ordered because the publication, uncertainty, and
attribution stages reuse outputs from earlier stages. Use
`python scripts/run_all.py --help` to run a smaller week range or skip notebook
execution.

## Individual Runs

All commands below assume the virtual environment is active and the current
directory is `S1_Linear/`.

### Week 3: UCI linear baselines

```bash
python scripts/run_week03_linear.py
python -m nbconvert --to notebook --execute --inplace notebooks/week03_linear_family.ipynb
```

The experiment uses an 80/20 train-test split and 4-fold cross-validation.

### Week 4: UCI feature representations

```bash
python scripts/run_week04_representation.py
python -m nbconvert --to notebook --execute --inplace notebooks/week04_representation_experiments.ipynb
```

This stage compares the original, domain-engineered, log-transformed, and
interaction feature sets. The selected setup is also evaluated with seeds
`0`, `42`, and `123`.

### Week 5: UHPC import and target audit

```bash
python -m nbconvert --to notebook --execute --inplace notebooks/week05_uhpc_import_and_target_check.ipynb
```

This notebook reads the UHPC workbook, selects the 28-day compressive-strength
target, and writes the target-filtered CSV used in the next stage.

### Week 6: UHPC semantic missingness

```bash
python scripts/run_week06_semantic_missingness.py
python -m nbconvert --to notebook --execute --inplace notebooks/week06_semantic_missingness_strategies.ipynb
```

This stage compares three missingness policies and fits preprocessing only on
training rows.

### Week 7: Shared UHPC row-mixed experiments

```bash
python scripts/run_week07_preprocess_shared_uhpc.py
python scripts/run_week07_linear_experiments.py
python -m nbconvert --to notebook --execute --inplace notebooks/week07_linear_family_results.ipynb
```

The grouped split contains 1,449 training, 311 validation, and 313 test rows.
The raw 33 predictors produce 60 transformed model columns.

### Week 8: Publication-held-out generalization

```bash
python -m s1_linear.week08.runner
python -m nbconvert --to notebook --execute --inplace notebooks/week08_publication_generalization.ipynb
```

This stage keeps publications separate across train, validation, and test,
then runs the publication-held-out and LOPO evaluations.

### Week 9: Uncertainty calibration

```bash
python -m s1_linear.week09.runner
python -m nbconvert --to notebook --execute --inplace notebooks/week09_uncertainty_calibration.ipynb
```

This stage evaluates Bayesian, bootstrap, and conformal prediction intervals
on the fixed publication split.

### Week 10: Feature attribution

```bash
python -m s1_linear.week10.runner
python -m nbconvert --to notebook --execute --inplace notebooks/week10_feature_attribution.ipynb
```

This stage computes original-feature and engineering-group attribution using
SHAP and held-out permutation importance.

## Outputs

Generated files are written to:

- `data/processed/` for intermediate datasets and split manifests
- `reports/tables/` for audit and result tables
- `reports/figures/` for plots
- `results/metrics/` for evaluation metrics
- `results/predictions/` for row-level predictions
- `results/models/` for fitted model artifacts

Most generated outputs are ignored by Git because they can be reproduced from
the included data, configuration files, and runners.
