"""Targeted Week 7 experiments for UHPC Linear Family models."""

import numpy as np
import pandas as pd

from s1_linear.week07_metrics import (
    evaluate_fitted_model,
    summarize_prediction_errors_by_group,
)
from s1_linear.week07_models import build_week07_models

EPSILON = 1e-9


def _matching_columns(X: pd.DataFrame, candidates: list[str]) -> list[str]:
    """Find columns by exact name or case-insensitive substring."""
    matches = []
    lowered = {column.casefold(): column for column in X.columns}

    for candidate in candidates:
        if candidate in X.columns and candidate not in matches:
            matches.append(candidate)
            continue

        candidate_lower = candidate.casefold()
        if candidate_lower in lowered and lowered[candidate_lower] not in matches:
            matches.append(lowered[candidate_lower])
            continue

        for column in X.columns:
            if candidate_lower in column.casefold() and column not in matches:
                matches.append(column)

    return matches


def _first_matching_column(X: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first available matching column."""
    matches = _matching_columns(X, candidates)
    return matches[0] if matches else None


def _to_numeric_frame(X: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert selected columns to numeric without mutating X."""
    return pd.DataFrame(
        {column: pd.to_numeric(X[column], errors="coerce") for column in columns},
        index=X.index,
    )


def detect_fiber_group(X: pd.DataFrame) -> pd.Series:
    """
    Detect whether fiber was used in each mix.

    Supports both the long Week 6 column names and the simplified teammate
    columns used in Week 7.
    """
    amount_columns = _matching_columns(
        X,
        [
            "Mix Constitutents | Fiber | Amount / Quantity of Fiber",
            "Mix Constitutents | Synergetic Fiber | Amount / Quantity of Fiber",
            "fiber1_amount",
            "fiber2_amount",
        ],
    )

    if not amount_columns:
        return pd.Series("fiber_amount_unknown", index=X.index, name="fiber_group")

    amounts = _to_numeric_frame(X, amount_columns)
    any_positive = amounts.gt(0).any(axis=1)
    all_known = amounts.notna().all(axis=1)
    total_amount = amounts.sum(axis=1, min_count=1)

    labels = pd.Series("fiber_amount_unknown", index=X.index, name="fiber_group")
    labels.loc[any_positive] = "fiber_used"
    labels.loc[all_known & total_amount.eq(0)] = "no_fiber_reported"
    return labels


def detect_curing_group(X: pd.DataFrame) -> pd.Series:
    """Detect broad curing regime groups from categorical or numeric columns."""
    categorical_candidates = [
        "curing_method",
        "curing regime",
        "curing method",
        "curing",
        "Curing Regime",
    ]
    categorical_columns = [
        column
        for column in _matching_columns(X, categorical_candidates)
        if not pd.api.types.is_numeric_dtype(X[column])
        and not any(
            token in column.casefold()
            for token in ["temperature", "temp", "humidity", "pressure"]
        )
    ]

    if categorical_columns:
        source = X[categorical_columns[0]].fillna("").astype(str).str.casefold()
        labels = pd.Series("other", index=X.index, name="curing_group")
        labels.loc[source.str.strip().eq("")] = "unknown_or_missing"
        labels.loc[source.str.contains("standard|normal", regex=True)] = (
            "standard_or_normal"
        )
        labels.loc[source.str.contains("heat|hot", regex=True)] = "heat"
        labels.loc[source.str.contains("steam", regex=False)] = "steam"
        labels.loc[source.str.contains("water", regex=False)] = "water"
        labels.loc[source.str.contains("autoclave", regex=False)] = "autoclave"
        return labels

    temperature_column = _first_matching_column(
        X,
        [
            "curing_temp",
            "Curing Regime | Temperature",
            "temperature",
        ],
    )
    if temperature_column is None:
        return pd.Series("unknown_or_missing", index=X.index, name="curing_group")

    temperature = pd.to_numeric(X[temperature_column], errors="coerce")
    labels = pd.Series("unknown_temperature", index=X.index, name="curing_group")
    labels.loc[temperature.le(30)] = "normal_temperature"
    labels.loc[temperature.gt(30)] = "elevated_temperature"
    return labels


def run_feature_policy_sensitivity(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize feature-policy sensitivity when multiple policies are available.

    Week 7 currently uses only the agreed 50% policy, so the runner does not
    treat this as a targeted experiment.
    """
    required = {"policy", "model", "split", "RMSE"}
    if metrics_df.empty or not required.issubset(metrics_df.columns):
        return pd.DataFrame()

    validation = metrics_df.query("split == 'validation'")
    test = metrics_df.query("split == 'test'")
    rows = []

    for model_name, model_validation in validation.groupby("model"):
        best_validation = model_validation.sort_values("RMSE").iloc[0]
        model_test = test[test["model"] == model_name]
        best_test = (
            model_test.sort_values("RMSE").iloc[0] if not model_test.empty else None
        )
        matching_test = model_test[model_test["policy"] == best_validation["policy"]]
        rows.append(
            {
                "model": model_name,
                "comparison_available": validation["policy"].nunique() > 1,
                "best_validation_policy": best_validation["policy"],
                "best_validation_RMSE": best_validation["RMSE"],
                "test_RMSE_for_best_validation_policy": (
                    matching_test["RMSE"].iloc[0] if not matching_test.empty else np.nan
                ),
                "best_test_policy": (
                    best_test["policy"] if best_test is not None else np.nan
                ),
                "best_test_RMSE": (
                    best_test["RMSE"] if best_test is not None else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def run_error_analysis_by_groups(
    best_prediction_df: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create strength-bin, fiber-group, and curing-group error summaries."""
    prediction_df = best_prediction_df.reset_index(drop=True)
    strength_bins = pd.cut(
        prediction_df["Actual"],
        bins=[-np.inf, 120, 150, 180, np.inf],
        labels=["<=120", "120-150", "150-180", ">180"],
    )
    strength_error = summarize_prediction_errors_by_group(
        prediction_df,
        strength_bins,
        "strength_bin",
    )
    fiber_error = summarize_prediction_errors_by_group(
        prediction_df,
        detect_fiber_group(X_test).reset_index(drop=True),
        "fiber_group",
    )
    curing_error = summarize_prediction_errors_by_group(
        prediction_df,
        detect_curing_group(X_test).reset_index(drop=True),
        "curing_group",
    )

    return strength_error, fiber_error, curing_error


def _feature_types(X_train: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Detect numeric and categorical feature names."""
    numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=["number"]).columns.tolist()
    return numeric_features, categorical_features


def _fit_and_evaluate_models(
    policy: str,
    experiment: str,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    config: dict,
    exploratory: bool = False,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    **extra,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit all models using the frozen parameters selected by training-only CV."""
    if numeric_features is None or categorical_features is None:
        numeric_features, categorical_features = _feature_types(X_train)
    models = build_week07_models(
        X_train=X_train,
        config=config,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    metrics_frames = []
    prediction_frames = []

    for model_name, model in models.items():
        fitted = model.fit(X_train, y_train)
        metrics, predictions = evaluate_fitted_model(
            model_name=model_name,
            model=fitted,
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            policy=policy,
            experiment=experiment,
            exploratory=exploratory,
            **extra,
        )
        metrics_frames.append(metrics)
        prediction_frames.append(predictions)

    return (
        pd.concat(metrics_frames, ignore_index=True),
        pd.concat(prediction_frames, ignore_index=True),
    )


def _remove_fiber_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Remove fiber and synergetic-fiber columns for ablation."""
    remove_columns = [
        column
        for column in X.columns
        if "| Fiber |" in column
        or "| Synergetic Fiber |" in column
        or column.casefold().startswith("fiber1_")
        or column.casefold().startswith("fiber2_")
    ]
    return X.drop(columns=remove_columns)


def run_fiber_ablation_experiment(
    policy,
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    config,
    numeric_features,
    categorical_features,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare models with all features versus models without fiber features."""
    variants = {
        "fiber_features_included": (X_train, X_val, X_test),
        "fiber_features_removed": (
            _remove_fiber_columns(X_train),
            _remove_fiber_columns(X_val),
            _remove_fiber_columns(X_test),
        ),
    }
    metrics_frames = []
    prediction_frames = []

    for experiment_name, (Xtr, Xv, Xte) in variants.items():
        variant_numeric = [column for column in numeric_features if column in Xtr]
        variant_categorical = [
            column for column in categorical_features if column in Xtr
        ]
        metrics, predictions = _fit_and_evaluate_models(
            policy=policy,
            experiment=experiment_name,
            X_train=Xtr,
            X_val=Xv,
            X_test=Xte,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            config=config,
            numeric_features=variant_numeric,
            categorical_features=variant_categorical,
            experiment_family="fiber_ablation",
        )
        metrics_frames.append(metrics)
        prediction_frames.append(predictions)

    return (
        pd.concat(metrics_frames, ignore_index=True),
        pd.concat(prediction_frames, ignore_index=True),
    )


def remove_target_outliers_from_training(
    X_train,
    y_train,
    lower_quantile,
    upper_quantile,
):
    """Remove target outliers from training only using train-derived thresholds."""
    lower_threshold = y_train.quantile(lower_quantile)
    upper_threshold = y_train.quantile(upper_quantile)
    keep_mask = y_train.between(lower_threshold, upper_threshold, inclusive="both")

    audit = {
        "lower_quantile": lower_quantile,
        "upper_quantile": upper_quantile,
        "lower_threshold": lower_threshold,
        "upper_threshold": upper_threshold,
        "training_rows_before": len(y_train),
        "training_rows_after": int(keep_mask.sum()),
        "training_rows_removed": int((~keep_mask).sum()),
    }

    return X_train.loc[keep_mask].copy(), y_train.loc[keep_mask].copy(), audit


def run_outlier_sensitivity_experiment(
    policy,
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    config,
    numeric_features,
    categorical_features,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare original training data with target-outlier-filtered training data."""
    outlier_config = config.get("experiments", {}).get("outliers", {})
    lower_quantile = outlier_config.get("lower_quantile", 0.025)
    upper_quantile = outlier_config.get("upper_quantile", 0.975)

    filtered_X_train, filtered_y_train, audit = remove_target_outliers_from_training(
        X_train,
        y_train,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
    )

    variants = {
        "original_training": (X_train, y_train, {}),
        "outliers_removed_from_training": (filtered_X_train, filtered_y_train, audit),
    }
    metrics_frames = []
    prediction_frames = []

    for experiment_name, (Xtr, ytr, extra) in variants.items():
        metrics, predictions = _fit_and_evaluate_models(
            policy=policy,
            experiment=experiment_name,
            X_train=Xtr,
            X_val=X_val,
            X_test=X_test,
            y_train=ytr,
            y_val=y_val,
            y_test=y_test,
            config=config,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            experiment_family="outlier_sensitivity",
            **extra,
        )
        metrics_frames.append(metrics)
        prediction_frames.append(predictions)

    return (
        pd.concat(metrics_frames, ignore_index=True),
        pd.concat(prediction_frames, ignore_index=True),
    )


def _amount_column(X: pd.DataFrame, aliases: list[str]) -> str | None:
    """Find one amount column from aliases."""
    return _first_matching_column(X, aliases)


def _numeric_series(X: pd.DataFrame, column: str | None) -> pd.Series | None:
    """Return a numeric series or None when the column is unavailable."""
    if column is None:
        return None
    return pd.to_numeric(X[column], errors="coerce")


def _sum_available(series_list: list[pd.Series]) -> pd.Series | None:
    """Sum available numeric series without treating unknown amounts as zero."""
    available = [series for series in series_list if series is not None]
    if not available:
        return None
    return pd.concat(available, axis=1).sum(axis=1, min_count=len(available))


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Create a finite ratio while avoiding division by zero."""
    valid = denominator.abs().gt(EPSILON)
    return numerator.where(valid) / denominator.where(valid)


def add_engineering_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Add exploratory UHPC ratio features without using the target.

    Raw amount columns used to build aggregate binder/SCM/fiber totals are
    removed afterward to avoid feeding both combined ingredients and their
    components to the same linear model.
    """
    engineered = X.copy()
    amount_aliases = {
        "cement": ["cement", "Cement Amount"],
        "silica_fume": ["silica_fume", "Silica Fume"],
        "fly_ash": ["fly_ash", "Flayash Amount", "Fly Ash Amount"],
        "slag": ["slag", "Slag Amount"],
        "ggbfs": ["ggbfs", "GGBFS"],
        "metakaolin": ["metakaolin", "Metakaolin"],
        "limestone_powder": ["limestone_powder", "Limestone Powder"],
        "quartz_powder": ["quartz_powder", "Quartzpowder", "Quartz powder"],
        "glass_powder": ["glass_powder", "Glass powder"],
        "rice_husk_ash": ["rice_husk_ash", "Rice husk ash"],
        "sand": ["sand", "Sand | Amount", "sand amount"],
        "water": ["water", "Water | Amount", "water amount"],
        "superplasticizer": ["sp_amount", "Superplasticizer | Amount"],
        "fiber1": ["fiber1_amount", "Fiber | Amount / Quantity of Fiber"],
        "fiber2": [
            "fiber2_amount",
            "Synergetic Fiber | Amount / Quantity of Fiber",
        ],
    }
    columns = {key: _amount_column(engineered, aliases) for key, aliases in amount_aliases.items()}
    series = {key: _numeric_series(engineered, column) for key, column in columns.items()}

    binder_keys = [
        "cement",
        "silica_fume",
        "fly_ash",
        "slag",
        "ggbfs",
        "metakaolin",
        "limestone_powder",
        "quartz_powder",
        "glass_powder",
        "rice_husk_ash",
    ]
    scm_keys = [key for key in binder_keys if key != "cement"]
    total_binder = _sum_available([series[key] for key in binder_keys])
    total_scm = _sum_available([series[key] for key in scm_keys])
    total_fiber = _sum_available([series["fiber1"], series["fiber2"]])

    if total_binder is not None:
        engineered["total_binder"] = total_binder
    if total_scm is not None:
        engineered["total_scm"] = total_scm
    if total_fiber is not None:
        engineered["total_fiber_amount"] = total_fiber

    ratio_inputs = {
        "water_binder_ratio": series["water"],
        "scm_binder_ratio": total_scm,
        "sand_binder_ratio": series["sand"],
        "superplasticizer_binder_ratio": series["superplasticizer"],
        "fiber_binder_ratio": total_fiber,
    }
    if total_binder is not None:
        for feature_name, numerator in ratio_inputs.items():
            if numerator is not None:
                engineered[feature_name] = _safe_ratio(numerator, total_binder)

    remove_component_columns = [
        column
        for key, column in columns.items()
        if column is not None
        and key
        in {
            "cement",
            "silica_fume",
            "fly_ash",
            "slag",
            "ggbfs",
            "metakaolin",
            "limestone_powder",
            "quartz_powder",
            "glass_powder",
            "rice_husk_ash",
            "fiber1",
            "fiber2",
        }
    ]
    return engineered.drop(columns=remove_component_columns)


def run_engineering_features_experiment(
    policy,
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    config,
    numeric_features,
    categorical_features,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare original features with exploratory engineered ratio features."""
    variants = {
        "original_features": (X_train, X_val, X_test),
        "engineered_ratio_features": (
            add_engineering_features(X_train),
            add_engineering_features(X_val),
            add_engineering_features(X_test),
        ),
    }
    metrics_frames = []
    prediction_frames = []

    for experiment_name, (Xtr, Xv, Xte) in variants.items():
        if experiment_name == "original_features":
            variant_numeric = numeric_features
            variant_categorical = categorical_features
        else:
            variant_numeric, variant_categorical = _feature_types(Xtr)
        metrics, predictions = _fit_and_evaluate_models(
            policy=policy,
            experiment=experiment_name,
            X_train=Xtr,
            X_val=Xv,
            X_test=Xte,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            config=config,
            exploratory=True,
            numeric_features=variant_numeric,
            categorical_features=variant_categorical,
            experiment_family="engineering_features",
        )
        metrics_frames.append(metrics)
        prediction_frames.append(predictions)

    return (
        pd.concat(metrics_frames, ignore_index=True),
        pd.concat(prediction_frames, ignore_index=True),
    )
