"""Matplotlib-only plots for Week 7 Linear Family interpretation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_COLORS = [
    "#2F6690",
    "#3A7D44",
    "#D97B29",
    "#8A5A9E",
    "#B44747",
    "#6B7280",
]


def _prepare_output_path(output_path: str | Path) -> Path:
    """Create the parent directory and return a Path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save_and_close(fig, output_path: str | Path) -> None:
    """Apply final layout, save a PNG, and close the figure."""
    path = _prepare_output_path(output_path)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _find_column(frame: pd.DataFrame, candidates: list[str]) -> str:
    """Find the first exact or case-insensitive matching column."""
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

    lowered = {column.casefold(): column for column in frame.columns}
    for candidate in candidates:
        if candidate.casefold() in lowered:
            return lowered[candidate.casefold()]

    raise KeyError(
        f"None of the required columns {candidates} were found. "
        f"Available columns: {frame.columns.tolist()}"
    )


def _test_rows(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Return test rows when a split column exists."""
    frame = metrics_df.copy()
    if "split" in frame.columns:
        frame = frame[frame["split"].astype(str).str.casefold().eq("test")]
    return frame


def _grouped_metric_plot(
    ax,
    frame: pd.DataFrame,
    metric: str,
    group_col: str,
    series_col: str,
    title: str,
) -> None:
    """Draw grouped bars for one metric."""
    grouped = (
        frame.groupby([group_col, series_col], dropna=False)[metric]
        .mean()
        .unstack(series_col)
    )
    if grouped.empty:
        raise ValueError(f"No rows are available to plot {metric}.")

    x_positions = np.arange(len(grouped.index))
    series_values = grouped.columns.tolist()
    width = min(0.8 / max(len(series_values), 1), 0.35)

    for index, series_value in enumerate(series_values):
        offset = (index - (len(series_values) - 1) / 2) * width
        ax.bar(
            x_positions + offset,
            grouped[series_value].to_numpy(),
            width=width,
            label=str(series_value),
            color=MODEL_COLORS[index % len(MODEL_COLORS)],
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(grouped.index.astype(str), rotation=25, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title=series_col.replace("_", " ").title(), fontsize=8)


def plot_metrics_comparison(metrics_df: pd.DataFrame, output_path: str | Path) -> None:
    """Plot test MAE and RMSE by model and policy."""
    frame = _test_rows(metrics_df)
    model_col = _find_column(frame, ["model", "model_name"])
    policy_col = _find_column(frame, ["policy"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    _grouped_metric_plot(
        axes[0],
        frame,
        "MAE",
        model_col,
        policy_col,
        "Test MAE by Model and Policy",
    )
    _grouped_metric_plot(
        axes[1],
        frame,
        "RMSE",
        model_col,
        policy_col,
        "Test RMSE by Model and Policy",
    )
    fig.suptitle("Week 7 Linear Family Test Error", fontsize=14)
    _save_and_close(fig, output_path)


def plot_r2_comparison(metrics_df: pd.DataFrame, output_path: str | Path) -> None:
    """Plot test R2 by model and policy."""
    frame = _test_rows(metrics_df)
    model_col = _find_column(frame, ["model", "model_name"])
    policy_col = _find_column(frame, ["policy"])

    fig, ax = plt.subplots(figsize=(8, 5))
    _grouped_metric_plot(
        ax,
        frame,
        "R2",
        model_col,
        policy_col,
        "Test R2 by Model and Policy",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    _save_and_close(fig, output_path)


def plot_predicted_vs_actual(
    prediction_df: pd.DataFrame,
    output_path: str | Path,
    title: str | None = None,
) -> None:
    """Plot actual against predicted strength with a perfect-fit reference."""
    actual_col = _find_column(prediction_df, ["Actual", "y_true", "actual"])
    predicted_col = _find_column(prediction_df, ["Predicted", "y_pred", "predicted"])
    frame = prediction_df[[actual_col, predicted_col]].apply(
        pd.to_numeric, errors="coerce"
    )
    frame = frame.dropna()

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(
        frame[actual_col],
        frame[predicted_col],
        alpha=0.65,
        edgecolor="white",
        linewidth=0.35,
        color="#2F6690",
    )
    lower = min(frame[actual_col].min(), frame[predicted_col].min())
    upper = max(frame[actual_col].max(), frame[predicted_col].max())
    ax.plot([lower, upper], [lower, upper], "--", color="#B44747", label="Perfect fit")
    ax.set_xlabel("Actual 28-day Compressive Strength (MPa)")
    ax.set_ylabel("Predicted 28-day Compressive Strength (MPa)")
    ax.set_title(title or "Predicted vs Actual Strength")
    ax.grid(alpha=0.2)
    ax.legend()
    _save_and_close(fig, output_path)


def plot_residuals(
    prediction_df: pd.DataFrame,
    output_path: str | Path,
    title: str | None = None,
) -> None:
    """Plot residuals against predicted strength."""
    predicted_col = _find_column(prediction_df, ["Predicted", "y_pred", "predicted"])
    frame = prediction_df.copy()
    try:
        residual_col = _find_column(frame, ["Residual", "residual"])
    except KeyError:
        actual_col = _find_column(frame, ["Actual", "y_true", "actual"])
        frame["Residual"] = pd.to_numeric(frame[actual_col], errors="coerce") - pd.to_numeric(
            frame[predicted_col], errors="coerce"
        )
        residual_col = "Residual"

    values = frame[[predicted_col, residual_col]].apply(pd.to_numeric, errors="coerce")
    values = values.dropna()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        values[predicted_col],
        values[residual_col],
        alpha=0.65,
        edgecolor="white",
        linewidth=0.35,
        color="#3A7D44",
    )
    ax.axhline(0, linestyle="--", color="#B44747", linewidth=1.2)
    ax.set_xlabel("Predicted 28-day Compressive Strength (MPa)")
    ax.set_ylabel("Residual: Actual - Predicted (MPa)")
    ax.set_title(title or "Residuals vs Predicted Strength")
    ax.grid(alpha=0.2)
    _save_and_close(fig, output_path)


def plot_error_distribution(
    prediction_df: pd.DataFrame,
    output_path: str | Path,
    title: str | None = None,
) -> None:
    """Plot the absolute-error distribution."""
    frame = prediction_df.copy()
    try:
        error_col = _find_column(frame, ["AbsoluteError", "absolute_error"])
    except KeyError:
        actual_col = _find_column(frame, ["Actual", "y_true", "actual"])
        predicted_col = _find_column(frame, ["Predicted", "y_pred", "predicted"])
        frame["AbsoluteError"] = (
            pd.to_numeric(frame[actual_col], errors="coerce")
            - pd.to_numeric(frame[predicted_col], errors="coerce")
        ).abs()
        error_col = "AbsoluteError"

    errors = pd.to_numeric(frame[error_col], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(errors, bins=20, color="#D97B29", edgecolor="white", alpha=0.85)
    ax.axvline(errors.median(), color="#2F6690", linestyle="--", label="Median")
    ax.set_xlabel("Absolute Error (MPa)")
    ax.set_ylabel("Number of Test Samples")
    ax.set_title(title or "Absolute Error Distribution")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    _save_and_close(fig, output_path)


def plot_error_by_group(
    error_group_df: pd.DataFrame,
    group_col: str,
    output_path: str | Path,
    title: str | None = None,
) -> None:
    """Plot group-level MAE and label bars with available sample counts."""
    if group_col not in error_group_df:
        raise KeyError(f"Group column '{group_col}' is not present.")

    mae_col = _find_column(error_group_df, ["MAE", "mae"])
    frame = error_group_df.copy()
    frame[mae_col] = pd.to_numeric(frame[mae_col], errors="coerce")
    frame = frame.dropna(subset=[mae_col]).sort_values(mae_col, ascending=False)

    fig, ax = plt.subplots(figsize=(max(7, len(frame) * 1.2), 5))
    bars = ax.bar(
        frame[group_col].astype(str),
        frame[mae_col],
        color="#8A5A9E",
        alpha=0.85,
    )
    ax.set_xlabel(group_col.replace("_", " ").title())
    ax.set_ylabel("MAE (MPa)")
    ax.set_title(title or f"Test MAE by {group_col.replace('_', ' ').title()}")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.2)

    count_col = next(
        (column for column in ["n_rows", "count", "n"] if column in frame.columns),
        None,
    )
    if count_col:
        for bar, count in zip(bars, frame[count_col]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"n={int(count)}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    _save_and_close(fig, output_path)


def plot_top_vif(
    vif_df: pd.DataFrame,
    output_path: str | Path,
    top_n: int = 15,
) -> None:
    """Plot the largest numeric-feature VIF values, including capped infinities."""
    feature_col = _find_column(vif_df, ["feature"])
    vif_col = _find_column(vif_df, ["VIF", "vif"])
    frame = vif_df[[feature_col, vif_col]].copy()
    frame[vif_col] = pd.to_numeric(frame[vif_col], errors="coerce")
    frame = frame.dropna(subset=[vif_col]).sort_values(vif_col, ascending=False).head(top_n)

    finite_values = frame.loc[np.isfinite(frame[vif_col]), vif_col]
    finite_max = finite_values.max() if not finite_values.empty else 10.0
    cap = max(float(finite_max) * 1.15, 10.0)
    frame["plot_vif"] = frame[vif_col].replace([np.inf, -np.inf], cap)
    frame = frame.sort_values("plot_vif")

    fig, ax = plt.subplots(figsize=(8, max(5, len(frame) * 0.42)))
    bars = ax.barh(frame[feature_col].astype(str), frame["plot_vif"], color="#B44747")
    ax.axvline(5, color="#D97B29", linestyle="--", linewidth=1, label="VIF = 5")
    ax.axvline(10, color="#8A5A9E", linestyle=":", linewidth=1, label="VIF = 10")
    ax.set_xlabel("Variance Inflation Factor")
    ax.set_title(f"Top {min(top_n, len(frame))} Numeric-Feature VIF Values")
    ax.grid(axis="x", alpha=0.2)
    ax.legend()

    for bar, original_value in zip(bars, frame[vif_col]):
        label = "inf" if np.isinf(original_value) else f"{original_value:.2f}"
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f" {label}",
            va="center",
            fontsize=8,
        )

    _save_and_close(fig, output_path)


def plot_experiment_comparison(
    metrics_df: pd.DataFrame,
    output_path: str | Path,
    experiment_name: str,
) -> None:
    """Compare test MAE and RMSE between variants of one experiment family."""
    frame = _test_rows(metrics_df)
    if "experiment_family" in frame.columns:
        frame = frame[frame["experiment_family"].astype(str).eq(experiment_name)]
    elif "experiment" in frame.columns:
        frame = frame[frame["experiment"].astype(str).eq(experiment_name)]

    if frame.empty:
        raise ValueError(f"No test metrics found for experiment '{experiment_name}'.")

    model_col = _find_column(frame, ["model", "model_name"])
    experiment_col = _find_column(frame, ["experiment"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _grouped_metric_plot(
        axes[0],
        frame,
        "MAE",
        model_col,
        experiment_col,
        f"{experiment_name.replace('_', ' ').title()}: Test MAE",
    )
    _grouped_metric_plot(
        axes[1],
        frame,
        "RMSE",
        model_col,
        experiment_col,
        f"{experiment_name.replace('_', ' ').title()}: Test RMSE",
    )
    _save_and_close(fig, output_path)


def plot_coefficient_importance(
    fitted_pipeline,
    output_path: str | Path,
    top_n: int = 20,
) -> None:
    """
    Plot the strongest positive and negative coefficients.

    Coefficients require careful interpretation when VIF is high because
    multicollinearity can make their signs and magnitudes unstable.
    """
    try:
        preprocessor = fitted_pipeline.named_steps["preprocessor"]
        model = fitted_pipeline.named_steps["model"]
        feature_names = np.asarray(preprocessor.get_feature_names_out(), dtype=str)
        coefficients = np.asarray(model.coef_, dtype=float).reshape(-1)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        print(f"Coefficient plot skipped: feature names or coefficients unavailable ({exc}).")
        return

    if len(feature_names) != len(coefficients):
        print(
            "Coefficient plot skipped: transformed feature-name count does not match "
            "the model coefficient count."
        )
        return

    coefficient_frame = pd.DataFrame(
        {"feature": feature_names, "coefficient": coefficients}
    )
    positive = coefficient_frame[coefficient_frame["coefficient"] > 0].nlargest(
        top_n, "coefficient"
    )
    negative = coefficient_frame[coefficient_frame["coefficient"] < 0].nsmallest(
        top_n, "coefficient"
    )
    selected = pd.concat([negative, positive]).sort_values("coefficient")

    if selected.empty:
        print("Coefficient plot skipped: the fitted model has no non-zero coefficients.")
        return

    colors = np.where(selected["coefficient"] >= 0, "#3A7D44", "#B44747")
    fig, ax = plt.subplots(figsize=(10, max(6, len(selected) * 0.3)))
    ax.barh(selected["feature"], selected["coefficient"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Model Coefficient")
    ax.set_title(
        f"Top Positive and Negative Coefficients "
        f"(up to {top_n} in each direction)"
    )
    ax.grid(axis="x", alpha=0.2)
    _save_and_close(fig, output_path)
