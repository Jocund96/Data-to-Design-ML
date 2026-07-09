"""Compare SHAP rankings with held-out permutation rankings."""

from __future__ import annotations

import pandas as pd


def _top_overlap(frame: pd.DataFrame, shap_rank: str, perm_rank: str, top_k: int) -> int:
    """Count entities appearing in both top-k rankings."""
    return int(frame[shap_rank].le(top_k).astype(bool).mul(frame[perm_rank].le(top_k)).sum())


def compare_feature_rankings(
    shap_feature_importance: pd.DataFrame,
    permutation_feature_importance: pd.DataFrame,
    shap_group_importance: pd.DataFrame,
    permutation_group_importance: pd.DataFrame,
    top_k_features: int,
    top_k_groups: int,
) -> pd.DataFrame:
    """Return feature-level and group-level agreement diagnostics."""
    feature = shap_feature_importance[
        ["original_feature", "shap_rank", "mean_abs_shap"]
    ].merge(
        permutation_feature_importance[
            ["original_feature", "permutation_rank", "mean_rmse_delta"]
        ],
        on="original_feature",
        how="inner",
        validate="one_to_one",
    )
    group = shap_group_importance[
        ["feature_group", "shap_rank", "mean_abs_shap"]
    ].merge(
        permutation_group_importance[
            ["feature_group", "permutation_rank", "mean_rmse_delta"]
        ],
        on="feature_group",
        how="inner",
        validate="one_to_one",
    )

    feature_corr = feature["shap_rank"].corr(
        feature["permutation_rank"],
        method="spearman",
    )
    group_corr = group["shap_rank"].corr(
        group["permutation_rank"],
        method="spearman",
    )
    rows = [
        {
            "level": "original_feature",
            "n_entities": len(feature),
            "top_k": top_k_features,
            "top_k_overlap": _top_overlap(
                feature,
                "shap_rank",
                "permutation_rank",
                top_k_features,
            ),
            "spearman_rank_correlation": feature_corr,
        },
        {
            "level": "feature_group",
            "n_entities": len(group),
            "top_k": top_k_groups,
            "top_k_overlap": _top_overlap(
                group,
                "shap_rank",
                "permutation_rank",
                top_k_groups,
            ),
            "spearman_rank_correlation": group_corr,
        },
    ]
    return pd.DataFrame(rows)

