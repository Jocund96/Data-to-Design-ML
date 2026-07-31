# S2 Kernel

This track applies kernel-based and instance-based regression methods (KNN, SVR, NuSVR) to predict concrete compressive strength. Work is organized by week, moving from a warm-up exercise through preprocessing, modelling, and uncertainty analysis on two datasets.

## Datasets

Located in `Datasets/`:

- **UCI Concrete Compressive Strength dataset** (`uci_concrete_data.csv`) — 1030 samples, 8 mix-component features (cement, slag, fly ash, water, superplasticizer, coarse/fine aggregate, age), target is compressive strength (MPa). Processed splits and engineered features are saved under `Datasets/processed/uci_concrete_dataset/`.
- **UHPC dataset** (`UHPC_dataset.csv`) — Ultra High Performance Concrete data pooled from many publications, ~2000 rows with a `paper_reference` column identifying the source publication and `cs_28d` (28-day compressive strength) as the target. Cleaned and recoded versions (raw, semantically recoded, at 20%/50% missingness thresholds) are saved under `Datasets/processed/uhpc_dataset/`.

## Weekly folders

- **Week-1** — Warm-up notebook (`california_housing.ipynb`) fitting a plain linear regression on the sklearn California Housing dataset to establish a baseline workflow.
- **Week-2-3-4** — EDA, imputation, and preprocessing of the UCI concrete dataset, followed by KNN/SVR/NuSVR modelling and comparison of engineered vs. non-engineered features.
- **Week-5-6-7-8** — Move to the UHPC dataset: cleaning and column standardization, semantic missingness recoding, KNN/SVR/NuSVR modelling with hyperparameter tuning across dataset variants, and targeted experiments and held-out-publication (LOPO) analysis. Best hyperparameters and comparisons are cached in `results.json` / `results/`.
- **Week-9-10** — Uncertainty quantification and interpretability on the UHPC dataset: conformal prediction intervals under random split, group split, and leave-one-group-out (by publication), plus SHAP and permutation-based feature importance analysis.

## Dependencies

Based on the imports used across these notebooks and helper modules:

- numpy
- pandas
- scipy
- scikit-learn
- matplotlib
- seaborn
- tqdm
- shap
- jupyter (to run the notebooks)
