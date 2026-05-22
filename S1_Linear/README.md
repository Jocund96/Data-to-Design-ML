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
