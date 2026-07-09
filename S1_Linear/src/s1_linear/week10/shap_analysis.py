"""Linear SHAP-style attribution for the frozen Week 9 Elastic Net."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ShapAttributionResult:
    """All SHAP-derived Week 10 attribution tables."""

    transformed_values: pd.DataFrame
    original_values: pd.DataFrame
    group_values: pd.DataFrame
    transformed_importance: pd.DataFrame
    original_importance: pd.DataFrame
    group_importance: pd.DataFrame
    prediction_audit: pd.DataFrame
    method: str
    shap_available: bool


def _dense(matrix) -> np.ndarray:
    """Convert sparse or dense matrices to a dense numpy array."""
    if hasattr(matrix, "toarray"):
        return np.asarray(matrix.toarray(), dtype=float)
    return np.asarray(matrix, dtype=float)


def _try_package_linear_shap(model, background, evaluation):
    """Use shap.LinearExplainer if shap is installed and compatible."""
    try:
        import shap  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local env
        return None, None, f"shap_unavailable:{type(exc).__name__}"

    try:  # pragma: no cover - depends on the optional shap runtime
        explainer = shap.LinearExplainer(model, background)
        explanation = explainer(evaluation)
        values = np.asarray(explanation.values, dtype=float)
        base_values = np.asarray(explanation.base_values, dtype=float)
        return values, base_values, "shap.LinearExplainer"
    except Exception as exc:
        return None, None, f"shap_failed:{type(exc).__name__}"


def compute_linear_shap_values(
    fitted_pipeline,
    X_background: pd.DataFrame,
    X_evaluation: pd.DataFrame,
    transformed_features: list[str],
    prefer_shap_package: bool = True,
) -> tuple[np.ndarray, np.ndarray, str, bool]:
    """
    Compute transformed-space SHAP values for a fitted linear pipeline.

    When the optional ``shap`` package is unavailable, the independent linear
    SHAP formula is exact for a linear model:

    ``phi_j = (x_j - E[x_j]) * beta_j``.
    """
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    model = fitted_pipeline.named_steps["model"]
    background = _dense(preprocessor.transform(X_background))
    evaluation = _dense(preprocessor.transform(X_evaluation))
    coefficients = np.asarray(model.coef_, dtype=float).reshape(-1)
    if evaluation.shape[1] != len(transformed_features):
        raise ValueError("Transformed feature names do not match transformed matrix.")
    if evaluation.shape[1] != len(coefficients):
        raise ValueError("Transformed feature count does not match coefficients.")

    if prefer_shap_package:
        values, base_values, method = _try_package_linear_shap(
            model,
            background,
            evaluation,
        )
        if values is not None and base_values is not None:
            return values, base_values, method, True

    background_mean = background.mean(axis=0)
    values = (evaluation - background_mean) * coefficients
    base_value = float(np.asarray(model.intercept_, dtype=float) + background_mean @ coefficients)
    base_values = np.full(evaluation.shape[0], base_value, dtype=float)
    return values, base_values, "independent_linear_formula", False


def _summarize_importance(
    values: pd.DataFrame,
    entity_column: str,
    extra: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarize row-level signed attributions into mean absolute importance."""
    rows = []
    for column in values.columns:
        if column == "row_position":
            continue
        series = pd.to_numeric(values[column], errors="coerce")
        rows.append(
            {
                entity_column: column,
                "mean_abs_shap": float(series.abs().mean()),
                "mean_signed_shap": float(series.mean()),
                "std_signed_shap": float(series.std(ddof=0)),
                "max_abs_shap": float(series.abs().max()),
            }
        )
    summary = pd.DataFrame(rows).sort_values(
        "mean_abs_shap",
        ascending=False,
    )
    total = summary["mean_abs_shap"].sum()
    summary["mean_abs_share"] = (
        summary["mean_abs_shap"] / total if total else np.nan
    )
    summary["shap_rank"] = np.arange(1, len(summary) + 1)
    if extra is not None:
        summary = summary.merge(extra, on=entity_column, how="left")
    return summary


def aggregate_shap_values(
    shap_values: np.ndarray,
    base_values: np.ndarray,
    fitted_pipeline,
    X_evaluation: pd.DataFrame,
    transformed_mapping: pd.DataFrame,
    feature_group_mapping: pd.DataFrame,
) -> ShapAttributionResult:
    """Aggregate transformed SHAP values to original features and groups."""
    transformed_features = transformed_mapping["transformed_feature"].tolist()
    transformed_values = pd.DataFrame(shap_values, columns=transformed_features)
    transformed_values.insert(0, "row_position", np.arange(len(transformed_values)))

    mapped = transformed_mapping.set_index("transformed_feature")
    original_map = mapped["original_feature"].to_dict()
    if mapped["original_feature"].isna().any():
        missing = mapped.loc[mapped["original_feature"].isna()].index.tolist()
        raise ValueError(f"Unmapped transformed features: {missing}")

    original_matrix = (
        transformed_values.drop(columns=["row_position"])
        .T.groupby(lambda name: original_map[name])
        .sum()
        .T
    )
    for feature in feature_group_mapping["original_feature"]:
        if feature not in original_matrix:
            original_matrix[feature] = 0.0
    original_matrix = original_matrix[
        feature_group_mapping["original_feature"].tolist()
    ]
    original_values = original_matrix.copy()
    original_values.insert(0, "row_position", np.arange(len(original_values)))

    group_map = feature_group_mapping.set_index("original_feature")[
        "feature_group"
    ].to_dict()
    group_matrix = original_matrix.T.groupby(lambda name: group_map[name]).sum().T
    group_values = group_matrix.copy()
    group_values.insert(0, "row_position", np.arange(len(group_values)))

    transformed_importance = _summarize_importance(
        transformed_values,
        "transformed_feature",
        transformed_mapping,
    )
    original_importance = _summarize_importance(
        original_values,
        "original_feature",
        feature_group_mapping,
    )
    group_importance = _summarize_importance(group_values, "feature_group")
    feature_abs_sum = (
        original_importance.groupby("feature_group", as_index=False)[
            "mean_abs_shap"
        ]
        .sum()
        .rename(columns={"mean_abs_shap": "sum_mean_abs_original_feature_shap"})
    )
    group_importance = group_importance.merge(
        feature_abs_sum,
        on="feature_group",
        how="left",
        validate="one_to_one",
    )

    prediction = np.asarray(fitted_pipeline.predict(X_evaluation), dtype=float)
    reconstructed = np.asarray(base_values, dtype=float) + shap_values.sum(axis=1)
    prediction_audit = pd.DataFrame(
        {
            "row_position": np.arange(len(prediction)),
            "prediction": prediction,
            "shap_base_value": base_values,
            "shap_sum": shap_values.sum(axis=1),
            "reconstructed_prediction": reconstructed,
            "absolute_additivity_error": np.abs(prediction - reconstructed),
        }
    )

    return ShapAttributionResult(
        transformed_values=transformed_values,
        original_values=original_values,
        group_values=group_values,
        transformed_importance=transformed_importance,
        original_importance=original_importance,
        group_importance=group_importance,
        prediction_audit=prediction_audit,
        method="",
        shap_available=False,
    )


def run_shap_attribution(
    fitted_pipeline,
    X_background: pd.DataFrame,
    X_evaluation: pd.DataFrame,
    transformed_mapping: pd.DataFrame,
    feature_group_mapping: pd.DataFrame,
    prefer_shap_package: bool,
) -> ShapAttributionResult:
    """Run transformed, original-feature, and group-level linear attribution."""
    transformed_features = transformed_mapping["transformed_feature"].tolist()
    values, base_values, method, shap_available = compute_linear_shap_values(
        fitted_pipeline=fitted_pipeline,
        X_background=X_background,
        X_evaluation=X_evaluation,
        transformed_features=transformed_features,
        prefer_shap_package=prefer_shap_package,
    )
    result = aggregate_shap_values(
        shap_values=values,
        base_values=base_values,
        fitted_pipeline=fitted_pipeline,
        X_evaluation=X_evaluation,
        transformed_mapping=transformed_mapping,
        feature_group_mapping=feature_group_mapping,
    )
    result.method = method
    result.shap_available = shap_available
    return result
