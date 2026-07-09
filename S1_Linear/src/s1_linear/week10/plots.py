"""Matplotlib plots for Week 10 feature attribution."""

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def _prepare_output(path: Path) -> None:
    """Create parent directories before saving a plot."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _barh(
    labels,
    values,
    output_path: Path,
    title: str,
    xlabel: str,
    color: str = "#3b73b9",
) -> None:
    _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.36)))
    positions = np.arange(len(labels))
    ax.barh(positions, values, color=color)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_top_original_shap(
    shap_importance: pd.DataFrame,
    output_path: Path,
    top_n: int = 15,
) -> None:
    """Plot top original features by mean absolute SHAP value."""
    data = shap_importance.sort_values("mean_abs_shap", ascending=False).head(top_n)
    _barh(
        labels=data["original_feature"],
        values=data["mean_abs_shap"],
        output_path=output_path,
        title="Week 10: Top Original Features by SHAP Magnitude",
        xlabel="Mean absolute SHAP contribution",
    )


def plot_group_shap(shap_group_importance: pd.DataFrame, output_path: Path) -> None:
    """Plot engineered feature groups by mean absolute signed group SHAP value."""
    data = shap_group_importance.sort_values("mean_abs_shap", ascending=False)
    _barh(
        labels=data["feature_group"],
        values=data["mean_abs_shap"],
        output_path=output_path,
        title="Week 10: Feature Groups by SHAP Magnitude",
        xlabel="Mean absolute group SHAP contribution",
        color="#4f9d69",
    )


def plot_feature_permutation(
    permutation_importance: pd.DataFrame,
    output_path: Path,
    top_n: int = 15,
) -> None:
    """Plot top original features by held-out RMSE increase after permutation."""
    data = permutation_importance.sort_values(
        "mean_rmse_delta",
        ascending=False,
    ).head(top_n)
    _barh(
        labels=data["original_feature"],
        values=data["mean_rmse_delta"],
        output_path=output_path,
        title="Week 10: Held-Out Feature Permutation Importance",
        xlabel="Mean RMSE increase after permutation",
        color="#b95f3b",
    )


def plot_group_permutation(
    permutation_importance: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot feature groups by held-out RMSE increase after group permutation."""
    data = permutation_importance.sort_values("mean_rmse_delta", ascending=False)
    _barh(
        labels=data["feature_group"],
        values=data["mean_rmse_delta"],
        output_path=output_path,
        title="Week 10: Held-Out Group Permutation Importance",
        xlabel="Mean RMSE increase after group permutation",
        color="#a65fb9",
    )


def plot_rank_agreement(
    shap_importance: pd.DataFrame,
    permutation_importance: pd.DataFrame,
    output_path: Path,
) -> None:
    """Scatter plot comparing SHAP rank and permutation rank."""
    data = shap_importance[
        ["original_feature", "shap_rank", "mean_abs_shap"]
    ].merge(
        permutation_importance[
            ["original_feature", "permutation_rank", "mean_rmse_delta"]
        ],
        on="original_feature",
        how="inner",
        validate="one_to_one",
    )
    _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(data["shap_rank"], data["permutation_rank"], alpha=0.75)
    max_rank = int(max(data["shap_rank"].max(), data["permutation_rank"].max()))
    ax.plot([1, max_rank], [1, max_rank], linestyle="--", color="black", alpha=0.5)
    for _, row in data.nsmallest(8, "shap_rank").iterrows():
        ax.annotate(
            row["original_feature"],
            (row["shap_rank"], row["permutation_rank"]),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_xlabel("SHAP rank (1 = most important)")
    ax.set_ylabel("Permutation rank (1 = most important)")
    ax.set_title("Week 10: SHAP vs Held-Out Permutation Rank")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

