# S1 Linear Family

This directory contains the Linear Family experiments for concrete
compressive-strength prediction. It begins with an introductory California
Housing example, followed by the UCI Concrete and literature-derived UHPC
datasets.

## View Saved Results

The notebooks are committed with their final saved outputs. They can be opened
directly on GitHub or in the VS Code Notebook Editor without rerunning the
experiments. In VS Code, install the Jupyter extension and trust the cloned
workspace so that rich tables and figures are displayed.

| Stage                                    | Notebook                                                         |
| ---------------------------------------- | ---------------------------------------------------------------- |
| Linear regression basics                 | [Week 1](notebooks/week01_linear_regression_basics.ipynb)         |
| UCI exploratory analysis                 | [Week 2](notebooks/week02_uci_exploratory_analysis.ipynb)         |
| UCI linear baselines                     | [Week 3](notebooks/week03_linear_family.ipynb)                   |
| UCI feature representations              | [Week 4](notebooks/week04_representation_experiments.ipynb)      |
| UHPC import and target audit             | [Week 5](notebooks/week05_uhpc_import_and_target_check.ipynb)    |
| UHPC semantic missingness                | [Week 6](notebooks/week06_semantic_missingness_strategies.ipynb) |
| UHPC row-mixed experiments               | [Week 7](notebooks/week07_linear_family_results.ipynb)           |
| Publication-held-out and LOPO evaluation | [Week 8](notebooks/week08_publication_generalization.ipynb)      |
| Uncertainty calibration                  | [Week 9](notebooks/week09_uncertainty_calibration.ipynb)         |
| SHAP and permutation attribution         | [Week 10](notebooks/week10_feature_attribution.ipynb)            |

Individual artifacts can also be inspected without execution:

- [Tables](reports/tables/) for preprocessing audits and experiment summaries
- [Figures](reports/figures/) for saved plots
- [Metrics](results/metrics/) for model-evaluation results
- [Predictions](results/predictions/) for row-level model outputs

## Requirements

The following environment is required only to rerun the experiments.

- Python 3.11 or newer
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
| UCI exploratory analysis         | `data/raw/uci_concrete_data.xlsx`                    |
| UCI baseline and representations | `data/processed/uci_concrete_clean_engineered.csv`   |
| UHPC import                      | `data/raw/UHPC Dataset  (Version-2).xlsx`            |
| UHPC semantic missingness        | `data/processed/uhpc_rows_with_28day_target.csv`     |
| Shared UHPC experiments          | `data/processed/shared_strategies/uhpc_semantic_50/` |

The UCI baseline uses the original eight Yeh predictors: cement, slag, fly
ash, water, superplasticizer, coarse aggregate, fine aggregate, and age.

## Optional Complete Run

The complete workflow runs the experiment code first and then executes each
notebook with the generated outputs:

```bash
python scripts/run_all.py
```

The workflows are ordered because the publication, uncertainty, and
attribution stages reuse outputs from earlier stages. Use
`python scripts/run_all.py --help` to run a smaller week range or skip notebook
execution.

## Optional Individual Runs

All commands below assume the virtual environment is active and the current
directory is `S1_Linear/`.

### Week 1: Linear regression basics

```bash
python -m nbconvert --to notebook --execute --inplace notebooks/week01_linear_regression_basics.ipynb
```

This notebook introduces data inspection, an 80/20 split, ordinary linear
regression, standard metrics, and an actual-versus-predicted plot using the
California Housing dataset retrieved through scikit-learn. On a machine where
the dataset is not already cached, its first execution requires internet access.

### Week 2: UCI exploratory analysis

```bash
python -m nbconvert --to notebook --execute --inplace notebooks/week02_uci_exploratory_analysis.ipynb
```

This notebook cleans the raw UCI workbook, examines missing values,
duplicates, distributions, correlations, and outliers, and creates the
processed raw and engineered feature tables used by the later experiments.

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

Readable tables, figures, metrics, predictions, and frozen YAML configurations
are included in the repository for direct inspection. Fitted model binaries are
ignored because they can be regenerated from the included data, configurations,
and runners.
