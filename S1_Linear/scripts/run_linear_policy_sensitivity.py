"""Compare the Linear family under the UHPC 20% and 50% feature policies."""

from __future__ import annotations

import json
from pathlib import Path
import sys

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, ElasticNet, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, TargetEncoder

from s1_linear.week07_metrics import regression_metrics
from s1_linear.week07_preprocessing import make_feature_hash_groups


TARGET = "Mechanical Properties | Compressive Strength (Mpa) | 28-Day"
POLICIES = ("raw_le_20", "raw_le_50")
STABLE_SELECTION_MODELS = ("Elastic Net", "Bayesian Ridge")
CV_FOLDS = 3
RANDOM_STATE = 42


def load_policy_splits(policy: str):
    """Load one fixed train/validation/test policy split."""
    split_dir = project_root / "data" / "processed" / "week6" / policy
    X_train = pd.read_csv(split_dir / "X_train.csv")
    X_val = pd.read_csv(split_dir / "X_val.csv")
    X_test = pd.read_csv(split_dir / "X_test.csv")
    y_train = pd.read_csv(split_dir / "y_train.csv")[TARGET]
    y_val = pd.read_csv(split_dir / "y_val.csv")[TARGET]
    y_test = pd.read_csv(split_dir / "y_test.csv")[TARGET]
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Build train-fitted mean imputation, encoding, and scaling."""
    numeric = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical = X_train.select_dtypes(exclude=["number"]).columns.tolist()

    transformers = [
        (
            "numeric",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric,
        )
    ]
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant",
                                fill_value="missing_reported_gap",
                            ),
                        ),
                        (
                            "encoder",
                            TargetEncoder(
                                cv=KFold(
                                    n_splits=5,
                                    shuffle=True,
                                    random_state=RANDOM_STATE,
                                ),
                                smooth="auto",
                                target_type="continuous",
                            ),
                        ),
                        ("scaler", StandardScaler()),
                    ]
                ),
                categorical,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_models(X_train: pd.DataFrame):
    """Return the four Linear models and their established tuning grids."""
    common = lambda model: Pipeline(
        [
            ("preprocessor", build_preprocessor(X_train)),
            ("model", model),
        ]
    )
    polynomial = Pipeline(
        [
            ("preprocessor", build_preprocessor(X_train)),
            ("poly", PolynomialFeatures(include_bias=False)),
            ("polynomial_scaler", StandardScaler()),
            (
                "model",
                Ridge(
                    solver="lsqr",
                    max_iter=5000,
                    tol=0.0001,
                ),
            ),
        ]
    )

    return {
        "OLS": (
            common(LinearRegression()),
            {},
        ),
        "Elastic Net": (
            common(
                ElasticNet(
                    max_iter=50000,
                    tol=0.0001,
                    random_state=RANDOM_STATE,
                )
            ),
            {
                "model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
                "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
            },
        ),
        "Bayesian Ridge": (
            common(BayesianRidge(max_iter=300, tol=0.001)),
            {
                "model__alpha_1": [1e-7, 1e-6, 1e-5],
                "model__lambda_1": [1e-7, 1e-6, 1e-5],
            },
        ),
        "Polynomial Ridge": (
            polynomial,
            {
                "poly__degree": [2],
                "model__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0],
            },
        ),
    }


def evaluate_policy(policy: str):
    """Tune and evaluate all Linear models for one policy."""
    X_train, X_val, X_test, y_train, y_val, y_test = load_policy_splits(policy)
    groups = make_feature_hash_groups(X_train)
    cv = GroupKFold(n_splits=CV_FOLDS)
    metrics_rows = []
    tuning_rows = []

    for model_name, (pipeline, grid) in build_models(X_train).items():
        if grid:
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=grid,
                scoring="neg_root_mean_squared_error",
                cv=cv,
                n_jobs=1,
                refit=True,
                return_train_score=True,
                error_score="raise",
            )
            search.fit(X_train, y_train, groups=groups)
            fitted = search.best_estimator_
            best_cv_rmse = -float(search.best_score_)
            best_parameters = search.best_params_
        else:
            fitted = pipeline.fit(X_train, y_train)
            best_cv_rmse = None
            best_parameters = {}

        tuning_rows.append(
            {
                "policy": policy,
                "model": model_name,
                "cv_folds": CV_FOLDS,
                "grouping": "feature_hash_group",
                "best_cv_RMSE": best_cv_rmse,
                "best_parameters": json.dumps(best_parameters, sort_keys=True),
            }
        )

        for split, X_split, y_split in (
            ("train", X_train, y_train),
            ("validation", X_val, y_val),
            ("test", X_test, y_test),
        ):
            metrics_rows.append(
                {
                    "policy": policy,
                    "model": model_name,
                    "split": split,
                    "numeric_imputer": "mean",
                    "categorical_encoder": "TargetEncoder(5-fold cross-fitting)",
                    **regression_metrics(y_split, fitted.predict(X_split)),
                }
            )

    return pd.DataFrame(metrics_rows), pd.DataFrame(tuning_rows)


def select_models(metrics: pd.DataFrame) -> pd.DataFrame:
    """Select the stable regularized model with the lowest validation RMSE."""
    rows = []
    for policy, policy_metrics in metrics.groupby("policy", sort=False):
        validation = policy_metrics[
            policy_metrics["model"].isin(STABLE_SELECTION_MODELS)
            & policy_metrics["split"].eq("validation")
        ].sort_values("RMSE")
        selected_validation = validation.iloc[0]
        selected_test = policy_metrics.query(
            "split == 'test' and model == @selected_validation.model"
        ).iloc[0]
        rows.append(
            {
                "policy": policy,
                "selection_rule": "lowest_validation_RMSE_stable_regularized",
                "selected_model": selected_validation["model"],
                "validation_MAE": selected_validation["MAE"],
                "validation_RMSE": selected_validation["RMSE"],
                "validation_R2": selected_validation["R2"],
                "test_MAE": selected_test["MAE"],
                "test_RMSE": selected_test["RMSE"],
                "test_R2": selected_test["R2"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Run both policies and save their comparison outputs."""
    metric_frames = []
    tuning_frames = []
    for policy in POLICIES:
        metrics, tuning = evaluate_policy(policy)
        metric_frames.append(metrics)
        tuning_frames.append(tuning)

    all_metrics = pd.concat(metric_frames, ignore_index=True)
    all_tuning = pd.concat(tuning_frames, ignore_index=True)
    selected = select_models(all_metrics)

    metrics_dir = project_root / "results" / "metrics"
    tables_dir = project_root / "reports" / "tables"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    all_metrics.to_csv(
        metrics_dir / "linear_policy_sensitivity_metrics.csv",
        index=False,
    )
    all_tuning.to_csv(
        tables_dir / "linear_policy_sensitivity_tuning.csv",
        index=False,
    )
    selected.to_csv(
        tables_dir / "linear_policy_sensitivity_selected.csv",
        index=False,
    )
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
