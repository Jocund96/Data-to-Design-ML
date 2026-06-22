"""Matplotlib plots for Week 8 publication-generalization analysis."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, output_path: str | Path) -> None:
    """Create the output directory, save the figure, and close it."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_publication_group_sizes(
    publication_audit: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Plot the distribution of usable rows per publication."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(publication_audit["n_rows"], bins=25, color="#2F6690", edgecolor="white")
    ax.axvline(
        publication_audit["n_rows"].median(),
        color="#B44747",
        linestyle="--",
        label=f"Median = {publication_audit['n_rows'].median():.0f}",
    )
    ax.set_xlabel("Usable Modeling Rows per Publication")
    ax.set_ylabel("Number of Publications")
    ax.set_title("Week 8 Publication Group-Size Distribution")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    _save(fig, output_path)


def plot_split_comparison(
    comparison: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Compare the same frozen model across row-mixed and publication-held-out tests."""
    frame = comparison.copy()
    labels = frame["split_strategy"].str.replace("_", " ").str.title()
    x = np.arange(len(frame))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].bar(x - width / 2, frame["MAE"], width, label="MAE", color="#3A7D44")
    axes[0].bar(x + width / 2, frame["RMSE"], width, label="RMSE", color="#B44747")
    axes[0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0].set_ylabel("Error (MPa)")
    axes[0].set_title("Same Frozen Model: Test Error")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend()

    axes[1].bar(labels, frame["R2"], color=["#2F6690", "#D97B29"])
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set_ylabel("R2")
    axes[1].set_title("Same Frozen Model: Test R2")
    axes[1].grid(axis="y", alpha=0.2)
    _save(fig, output_path)


def plot_unseen_predicted_actual(
    predictions: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Plot predictions for the final unseen-publication test."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(
        predictions["Actual"],
        predictions["Predicted"],
        alpha=0.65,
        color="#2F6690",
        edgecolor="white",
        linewidth=0.3,
    )
    lower = min(predictions["Actual"].min(), predictions["Predicted"].min())
    upper = max(predictions["Actual"].max(), predictions["Predicted"].max())
    ax.plot([lower, upper], [lower, upper], "--", color="#B44747")
    ax.set_xlabel("Actual 28-day Strength (MPa)")
    ax.set_ylabel("Predicted 28-day Strength (MPa)")
    ax.set_title("Final Test: Completely Unseen Publications")
    ax.grid(alpha=0.2)
    _save(fig, output_path)


def plot_lopo_rmse_ranking(
    lopo_metrics: pd.DataFrame,
    output_path: str | Path,
    top_n: int = 30,
) -> None:
    """Rank the worst leave-one-publication-out RMSE values."""
    frame = lopo_metrics.nlargest(top_n, "RMSE").sort_values("RMSE")
    fig, ax = plt.subplots(figsize=(9, max(6, len(frame) * 0.3)))
    bars = ax.barh(frame["publication_group"], frame["RMSE"], color="#B44747")
    for bar, rows in zip(bars, frame["n_rows"]):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" n={rows}", va="center", fontsize=7)
    ax.set_xlabel("Leave-One-Publication-Out RMSE (MPa)")
    ax.set_title(f"Worst {len(frame)} Publications by RMSE")
    ax.grid(axis="x", alpha=0.2)
    _save(fig, output_path)


def plot_lopo_bias(
    lopo_metrics: pd.DataFrame,
    output_path: str | Path,
    top_n: int = 30,
) -> None:
    """Plot publications with the largest absolute systematic bias."""
    frame = lopo_metrics.assign(abs_bias=lopo_metrics["Bias"].abs()).nlargest(
        top_n,
        "abs_bias",
    )
    frame = frame.sort_values("Bias")
    colors = np.where(frame["Bias"] >= 0, "#D97B29", "#2F6690")
    fig, ax = plt.subplots(figsize=(9, max(6, len(frame) * 0.3)))
    ax.barh(frame["publication_group"], frame["Bias"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Bias: Actual - Predicted (MPa)")
    ax.set_title("Largest Publication-Level Bias")
    ax.grid(axis="x", alpha=0.2)
    _save(fig, output_path)


def plot_shift_diagnostics(
    lopo_metrics: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Relate publication RMSE to numeric-range and unseen-category shift."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(
        lopo_metrics["numeric_out_of_training_range_rate"] * 100,
        lopo_metrics["RMSE"],
        s=np.maximum(lopo_metrics["n_rows"], 5) * 2,
        alpha=0.65,
        color="#3A7D44",
    )
    axes[0].set_xlabel("Numeric Values Outside Training Range (%)")
    axes[0].set_ylabel("LOPO RMSE (MPa)")
    axes[0].set_title("Error vs Numeric Range Shift")
    axes[0].grid(alpha=0.2)

    axes[1].scatter(
        lopo_metrics["unseen_category_rate"] * 100,
        lopo_metrics["RMSE"],
        s=np.maximum(lopo_metrics["n_rows"], 5) * 2,
        alpha=0.65,
        color="#8A5A9E",
    )
    axes[1].set_xlabel("Unseen Categorical Values (%)")
    axes[1].set_ylabel("LOPO RMSE (MPa)")
    axes[1].set_title("Error vs Category Shift")
    axes[1].grid(alpha=0.2)
    _save(fig, output_path)
