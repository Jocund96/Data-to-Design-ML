"""Leakage-safe hyperparameter tuning utilities for Week 7 Linear models."""

from copy import deepcopy
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, GroupKFold

from s1_linear.week07_models import MODEL_NAME_TO_CONFIG_KEY, build_week07_models
from s1_linear.week07_preprocessing import make_feature_hash_groups


def _python_value(value):
    """Convert numpy scalar values into JSON-safe Python values."""
    return value.item() if isinstance(value, np.generic) else value


def _json_params(params: dict) -> str:
    """Serialize a parameter dictionary consistently for report tables."""
    return json.dumps(
        {key: _python_value(value) for key, value in params.items()},
        sort_keys=True,
    )


def make_frozen_model_config(
    config: dict,
    best_pipeline_params: dict[str, dict],
) -> dict:
    """Copy config and apply tuned pipeline parameters as fixed model settings."""
    frozen = deepcopy(config)

    for model_name, pipeline_params in best_pipeline_params.items():
        model_key = MODEL_NAME_TO_CONFIG_KEY[model_name]
        model_config = frozen.setdefault("models", {}).setdefault(model_key, {})

        for parameter, value in pipeline_params.items():
            if parameter.startswith("model__"):
                model_config[parameter.removeprefix("model__")] = _python_value(value)
            elif parameter.startswith("poly__"):
                model_config[parameter.removeprefix("poly__")] = _python_value(value)

    frozen["hyperparameter_tuning"]["enabled"] = False
    frozen["hyperparameter_tuning"]["frozen_from_training_cv"] = True
    return frozen


def _compact_cv_results(model_name: str, grid: GridSearchCV) -> pd.DataFrame:
    """Create a readable candidate-level CV result table."""
    raw = pd.DataFrame(grid.cv_results_)
    compact = pd.DataFrame(
        {
            "model": model_name,
            "candidate_parameters": raw["params"].map(_json_params),
            "rank_cv_RMSE": raw["rank_test_score"].astype(int),
            "mean_cv_RMSE": -raw["mean_test_score"],
            "std_cv_RMSE": raw["std_test_score"],
            "mean_train_cv_RMSE": -raw["mean_train_score"],
            "mean_fit_time_seconds": raw["mean_fit_time"],
            "std_fit_time_seconds": raw["std_fit_time"],
        }
    )
    return compact.sort_values(["rank_cv_RMSE", "mean_cv_RMSE"])


def tune_week07_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    groups: pd.Series | None = None,
    grouping_name: str = "feature_hash_group",
) -> tuple[dict, pd.DataFrame, pd.DataFrame, dict]:
    """
    Tune enabled models using group-aware CV on training rows only.

    Preprocessing remains inside each pipeline, so every CV fold fits its own
    imputer, encoder, scaler, polynomial expansion, and model.
    """
    tuning_config = config.get("hyperparameter_tuning", {})
    if not tuning_config.get("enabled", False):
        raise ValueError("Week 7 hyperparameter tuning is not enabled in config.")

    models = build_week07_models(
        X_train=X_train,
        config=config,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    if groups is None:
        groups = make_feature_hash_groups(X_train)
    else:
        groups = pd.Series(groups).reset_index(drop=True)
        if len(groups) != len(X_train):
            raise ValueError("CV group labels must align with X_train rows.")
        if groups.isna().any():
            raise ValueError("CV group labels cannot contain missing values.")
    cv_folds = int(tuning_config.get("cv_folds", 3))
    if groups.nunique() < cv_folds:
        raise ValueError("Not enough feature-hash groups for the configured CV folds.")

    cv = GroupKFold(n_splits=cv_folds)
    grids = tuning_config.get("grids", {})
    scoring = tuning_config.get("scoring", "neg_root_mean_squared_error")
    n_jobs = int(tuning_config.get("n_jobs", 1))
    verbose = int(tuning_config.get("verbose", 1))

    fitted_models = {}
    best_params_by_model = {}
    summary_rows = []
    cv_frames = []

    for model_name, pipeline in models.items():
        model_key = MODEL_NAME_TO_CONFIG_KEY[model_name]
        parameter_grid = grids.get(model_key, {})

        if not parameter_grid:
            fitted_models[model_name] = pipeline.fit(X_train, y_train)
            best_params_by_model[model_name] = {}
            summary_rows.append(
                {
                    "model": model_name,
                    "tuned": False,
                    "cv_strategy": f"GroupKFold(n_splits={cv_folds})",
                    "grouping": grouping_name,
                    "scoring": scoring,
                    "candidate_count": 1,
                    "best_cv_RMSE": np.nan,
                    "best_parameters": "{}",
                    "note": "No meaningful hyperparameters; fitted as reference baseline.",
                }
            )
            continue

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            refit=True,
            return_train_score=True,
            error_score="raise",
            verbose=verbose,
        )
        grid.fit(X_train, y_train, groups=groups)
        fitted_models[model_name] = grid.best_estimator_
        best_params_by_model[model_name] = grid.best_params_
        cv_frames.append(_compact_cv_results(model_name, grid))
        summary_rows.append(
            {
                "model": model_name,
                "tuned": True,
                "cv_strategy": f"GroupKFold(n_splits={cv_folds})",
                "grouping": grouping_name,
                "scoring": scoring,
                "candidate_count": len(grid.cv_results_["params"]),
                "best_cv_RMSE": -float(grid.best_score_),
                "best_parameters": _json_params(grid.best_params_),
                "note": "Best estimator refitted on all supplied training rows.",
            }
        )

    frozen_config = make_frozen_model_config(config, best_params_by_model)
    cv_results = (
        pd.concat(cv_frames, ignore_index=True) if cv_frames else pd.DataFrame()
    )
    return fitted_models, pd.DataFrame(summary_rows), cv_results, frozen_config
