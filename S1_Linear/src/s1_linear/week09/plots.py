"""Matplotlib plots for Week 9 uncertainty and calibration results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_LABELS = {
    "elastic_net_split_conformal": "Elastic Net\nsplit conformal",
    "bayesian_ridge_native": "Bayesian Ridge\nnative",
    "bayesian_ridge_conformalized": "Bayesian Ridge\nconformalized",
    "elastic_net_residual_bootstrap": "Elastic Net\nbootstrap",
    "elastic_net_bootstrap_conformalized": "Elastic Net\nbootstrap + conformal",
}


def _prepare_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _labels(methods) -> list[str]:
    return [METHOD_LABELS.get(method, method.replace("_", " ")) for method in methods]


def plot_method_coverage_width(metrics_df: pd.DataFrame, output_path) -> None:
    """Compare primary empirical coverage and mean width by method."""
    frame = metrics_df.sort_values("EmpiricalCoverage", ascending=False)
    x = np.arange(len(frame))
    colors = plt.cm.Blues(np.linspace(0.45, 0.9, len(frame)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    axes[0].bar(x, frame["EmpiricalCoverage"] * 100, color=colors)
    axes[0].axhline(
        frame["nominal_coverage"].iloc[0] * 100,
        color="black",
        linestyle="--",
        label="Nominal coverage",
    )
    axes[0].set_ylabel("Empirical coverage (%)")
    axes[0].set_ylim(0, 105)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, frame["MeanIntervalWidth"], color=colors)
    axes[1].set_ylabel("Mean interval width (MPa)")
    axes[1].grid(axis="y", alpha=0.25)
    for axis in axes:
        axis.set_xticks(x, _labels(frame["method"]), rotation=18, ha="right")
    fig.suptitle("Week 9 Shared Publication-Held-Out Calibration")
    fig.tight_layout()
    fig.savefig(_prepare_path(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_coverage_curve(coverage_curve_df: pd.DataFrame, output_path) -> None:
    """Plot empirical coverage against requested nominal coverage."""
    fig, ax = plt.subplots(figsize=(8, 7))
    for method, group in coverage_curve_df.groupby("method", sort=False):
        group = group.sort_values("nominal_coverage")
        ax.plot(
            group["nominal_coverage"] * 100,
            group["EmpiricalCoverage"] * 100,
            marker="o",
            linewidth=2,
            label=METHOD_LABELS.get(method, method),
        )
    levels = np.linspace(45, 100, 100)
    ax.plot(levels, levels, color="black", linestyle="--", label="Ideal calibration")
    ax.set_xlabel("Nominal coverage (%)")
    ax.set_ylabel("Empirical coverage (%)")
    ax.set_xlim(45, 100)
    ax.set_ylim(45, 100)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    ax.set_title("Shared-Test Coverage Calibration Curve")
    fig.tight_layout()
    fig.savefig(_prepare_path(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_publication_coverage(
    publication_metrics_df: pd.DataFrame,
    output_path,
    method: str = "elastic_net_split_conformal",
) -> None:
    """Rank shared-test publications by empirical coverage."""
    frame = publication_metrics_df.loc[
        publication_metrics_df["method"].eq(method)
    ].sort_values("EmpiricalCoverage")
    labels = [f"{group} (n={int(rows)})" for group, rows in zip(frame["publication_group"], frame["n_rows"])]
    fig_height = max(6, len(frame) * 0.34)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    y = np.arange(len(frame))
    ax.barh(y, frame["EmpiricalCoverage"] * 100, color="#3A7CA5")
    ax.axvline(
        frame["nominal_coverage"].iloc[0] * 100,
        color="black",
        linestyle="--",
        label="Nominal coverage",
    )
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Empirical coverage (%)")
    ax.set_title(f"Publication Coverage: {METHOD_LABELS.get(method, method)}")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(_prepare_path(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_coverage_vs_width(publication_metrics_df: pd.DataFrame, output_path) -> None:
    """Show publication coverage and sharpness together."""
    fig, ax = plt.subplots(figsize=(10, 7))
    for method, group in publication_metrics_df.groupby("method", sort=False):
        ax.scatter(
            group["MeanIntervalWidth"],
            group["EmpiricalCoverage"] * 100,
            s=np.maximum(group["n_rows"], 5) * 2,
            alpha=0.65,
            label=METHOD_LABELS.get(method, method),
        )
    ax.axhline(
        publication_metrics_df["nominal_coverage"].iloc[0] * 100,
        color="black",
        linestyle="--",
        label="Nominal coverage",
    )
    ax.set_xlabel("Mean interval width (MPa)")
    ax.set_ylabel("Publication coverage (%)")
    ax.set_title("Publication Coverage Versus Interval Width")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(_prepare_path(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_interval_examples(
    predictions_df: pd.DataFrame,
    publication_metrics_df: pd.DataFrame,
    output_path,
    method: str = "elastic_net_split_conformal",
    max_publications: int = 3,
) -> None:
    """Plot intervals for the lowest-coverage publications."""
    metrics = publication_metrics_df.loc[
        publication_metrics_df["method"].eq(method)
    ].nsmallest(max_publications, ["EmpiricalCoverage", "MeanIntervalWidth"])
    groups = metrics["publication_group"].tolist()
    if not groups:
        return
    fig, axes = plt.subplots(len(groups), 1, figsize=(12, 4 * len(groups)), squeeze=False)
    for axis, publication_group in zip(axes[:, 0], groups):
        group = predictions_df.loc[
            predictions_df["method"].eq(method)
            & predictions_df["publication_group"].eq(publication_group)
        ].sort_values("Actual").reset_index(drop=True)
        x = np.arange(len(group))
        axis.fill_between(x, group["Lower"], group["Upper"], color="#A8DADC", alpha=0.7, label="90% interval")
        axis.plot(x, group["Predicted"], color="#1D3557", linewidth=2, label="Prediction")
        axis.scatter(x, group["Actual"], color="#E63946", s=25, label="Actual")
        axis.set_title(f"{publication_group}: coverage {group['Covered'].mean() * 100:.1f}% (n={len(group)})")
        axis.set_ylabel("28-day strength (MPa)")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    axes[-1, 0].set_xlabel("Rows sorted by actual strength")
    fig.suptitle(f"Lowest-Coverage Shared-Test Publications: {METHOD_LABELS.get(method, method)}", y=1.01)
    fig.tight_layout()
    fig.savefig(_prepare_path(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_lopo_coverage(
    lopo_publication_metrics: pd.DataFrame,
    output_path,
    method: str = "elastic_net_split_conformal",
) -> None:
    """Plot 90% interval coverage for the six thresholded LOPO publications."""
    frame = lopo_publication_metrics.loc[
        lopo_publication_metrics["method"].eq(method)
    ].sort_values("EmpiricalCoverage")
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(frame))
    ax.bar(x, frame["EmpiricalCoverage"] * 100, color="#457B9D")
    ax.axhline(
        frame["nominal_coverage"].iloc[0] * 100,
        color="black",
        linestyle="--",
        label="Nominal coverage",
    )
    ax.set_xticks(
        x,
        [f"{group}\n(n={int(rows)})" for group, rows in zip(frame["publication_group"], frame["n_rows"])],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("Empirical coverage (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Thresholded LOPO Coverage: Elastic Net Split Conformal")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(_prepare_path(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_uncertainty_vs_error(predictions_df: pd.DataFrame, output_path) -> None:
    """Compare adaptive uncertainty scores with absolute prediction errors."""
    adaptive = predictions_df.loc[
        predictions_df["method"].isin(
            [
                "bayesian_ridge_native",
                "bayesian_ridge_conformalized",
                "elastic_net_residual_bootstrap",
                "elastic_net_bootstrap_conformalized",
            ]
        )
    ]
    fig, ax = plt.subplots(figsize=(10, 7))
    for method, group in adaptive.groupby("method", sort=False):
        ax.scatter(
            group["UncertaintyScore"],
            group["AbsoluteError"],
            alpha=0.35,
            s=22,
            label=METHOD_LABELS.get(method, method),
        )
    ax.set_xlabel("Uncertainty score")
    ax.set_ylabel("Absolute error (MPa)")
    ax.set_title("Does Larger Model Uncertainty Track Larger Error?")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(_prepare_path(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_lopo_shift_vs_uncertainty(comparison_df: pd.DataFrame, output_path) -> None:
    """Relate Week 8 shift diagnostics to Week 9 LOPO interval width."""
    frame = comparison_df.loc[
        comparison_df["method"].eq("bayesian_ridge_conformalized")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    diagnostics = [
        ("numeric_out_of_training_range_rate", "Numeric out-of-range rate (%)"),
        ("unseen_category_rate", "Unseen-category rate (%)"),
    ]
    for axis, (column, label) in zip(axes, diagnostics):
        axis.scatter(
            frame[column] * 100,
            frame["MeanIntervalWidth"],
            s=np.maximum(frame["n_rows"], 5) * 2,
            color="#2A9D8F",
        )
        for _, row in frame.iterrows():
            axis.annotate(
                row["publication_group"].replace("-Research", ""),
                (row[column] * 100, row["MeanIntervalWidth"]),
                fontsize=7,
                xytext=(3, 3),
                textcoords="offset points",
            )
        axis.set_xlabel(label)
        axis.set_ylabel("Mean interval width (MPa)")
        axis.grid(alpha=0.25)
    fig.suptitle("LOPO Shift Diagnostics Versus Adaptive Interval Width")
    fig.tight_layout()
    fig.savefig(_prepare_path(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)

