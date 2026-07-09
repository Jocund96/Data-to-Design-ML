"""Held-out permutation importance for Week 10 feature attribution."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(y_true, y_pred) -> float:
    """Root mean squared error."""
    actual = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((actual - prediction) ** 2)))


def _shuffle_columns(
    X: pd.DataFrame,
    columns: list[str],
    permutation: np.ndarray,
) -> pd.DataFrame:
    """Return a copy of X with selected columns permuted by one row order."""
    shuffled = X.copy()
    shuffled.loc[:, columns] = shuffled.loc[:, columns].iloc[permutation].to_numpy()
    return shuffled


def _summarize_repeats(
    repeat_df: pd.DataFrame,
    entity_column: str,
    baseline_rmse: float,
) -> pd.DataFrame:
    """Summarize repeated permutation scores."""
    summary = (
        repeat_df.groupby(entity_column)
        .agg(
            n_repeats=("repeat", "count"),
            baseline_rmse=("baseline_rmse", "first"),
            mean_permuted_rmse=("permuted_rmse", "mean"),
            std_permuted_rmse=("permuted_rmse", "std"),
            mean_rmse_delta=("rmse_delta", "mean"),
            std_rmse_delta=("rmse_delta", "std"),
            min_rmse_delta=("rmse_delta", "min"),
            max_rmse_delta=("rmse_delta", "max"),
        )
        .reset_index()
    )
    summary["std_permuted_rmse"] = summary["std_permuted_rmse"].fillna(0.0)
    summary["std_rmse_delta"] = summary["std_rmse_delta"].fillna(0.0)
    summary["mean_rmse_delta_percentage"] = (
        summary["mean_rmse_delta"] / baseline_rmse * 100 if baseline_rmse else np.nan
    )
    summary = summary.sort_values("mean_rmse_delta", ascending=False)
    summary["permutation_rank"] = np.arange(1, len(summary) + 1)
    return summary


def run_feature_permutation_importance(
    fitted_pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    features: list[str],
    n_repeats: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shuffle one original feature at a time on the held-out test set."""
    if n_repeats < 1:
        raise ValueError("Permutation importance requires at least one repeat.")
    rng = np.random.default_rng(random_state)
    baseline_prediction = fitted_pipeline.predict(X_test)
    baseline = rmse(y_test, baseline_prediction)
    rows = []
    for feature in features:
        if feature not in X_test:
            continue
        for repeat in range(n_repeats):
            permutation = rng.permutation(len(X_test))
            shuffled = _shuffle_columns(X_test, [feature], permutation)
            score = rmse(y_test, fitted_pipeline.predict(shuffled))
            rows.append(
                {
                    "original_feature": feature,
                    "repeat": repeat,
                    "baseline_rmse": baseline,
                    "permuted_rmse": score,
                    "rmse_delta": score - baseline,
                }
            )
    repeats = pd.DataFrame(rows)
    summary = _summarize_repeats(repeats, "original_feature", baseline)
    return summary, repeats


def run_group_permutation_importance(
    fitted_pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_group_mapping: pd.DataFrame,
    n_repeats: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shuffle all original features in a group together on held-out test rows."""
    if n_repeats < 1:
        raise ValueError("Permutation importance requires at least one repeat.")
    rng = np.random.default_rng(random_state)
    baseline_prediction = fitted_pipeline.predict(X_test)
    baseline = rmse(y_test, baseline_prediction)
    rows = []
    for group, group_rows in feature_group_mapping.groupby("feature_group"):
        features = [
            feature
            for feature in group_rows["original_feature"].tolist()
            if feature in X_test
        ]
        if not features:
            continue
        for repeat in range(n_repeats):
            permutation = rng.permutation(len(X_test))
            shuffled = _shuffle_columns(X_test, features, permutation)
            score = rmse(y_test, fitted_pipeline.predict(shuffled))
            rows.append(
                {
                    "feature_group": group,
                    "features_permuted": ",".join(features),
                    "n_features_permuted": len(features),
                    "repeat": repeat,
                    "baseline_rmse": baseline,
                    "permuted_rmse": score,
                    "rmse_delta": score - baseline,
                }
            )
    repeats = pd.DataFrame(rows)
    summary = _summarize_repeats(repeats, "feature_group", baseline)
    group_sizes = (
        repeats.groupby("feature_group", as_index=False)
        .agg(
            features_permuted=("features_permuted", "first"),
            n_features_permuted=("n_features_permuted", "first"),
        )
    )
    summary = summary.merge(
        group_sizes,
        on="feature_group",
        how="left",
        validate="one_to_one",
    )
    return summary, repeats

