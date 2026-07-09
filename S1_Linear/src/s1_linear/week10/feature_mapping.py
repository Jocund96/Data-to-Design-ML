"""Feature-name and engineering-group mapping for Week 10 attribution."""

from __future__ import annotations

import pandas as pd


def make_feature_group_mapping(
    raw_features: list[str],
    configured_groups: dict[str, list[str]],
) -> pd.DataFrame:
    """Map each raw feature to one engineered interpretation group."""
    rows = []
    assigned = {}
    for group, features in configured_groups.items():
        for feature in features:
            assigned[feature] = group

    for feature in raw_features:
        group = assigned.get(feature, "other_unmapped")
        rows.append(
            {
                "original_feature": feature,
                "feature_group": group,
                "mapping_status": "configured" if feature in assigned else "unmapped",
            }
        )
    return pd.DataFrame(rows)


def _match_one_hot_source(remainder: str, original_features: list[str]) -> str | None:
    """Infer the original source column from a verbose one-hot feature name."""
    for feature in sorted(original_features, key=len, reverse=True):
        if remainder == feature or remainder.startswith(f"{feature}_"):
            return feature
    return None


def map_transformed_to_original(
    transformed_features: list[str],
    original_features: list[str],
) -> pd.DataFrame:
    """
    Map transformed model columns back to original shared predictors.

    The Week 7+ shared preprocessor creates names such as ``num__cement``,
    ``target__cement_type``, and ``ohe__curing_method_Water Curing``.
    """
    rows = []
    original_set = set(original_features)
    for transformed in transformed_features:
        if "__" in transformed:
            transform_family, remainder = transformed.split("__", 1)
        else:
            transform_family, remainder = "unknown", transformed

        if transform_family in {"num", "target"} and remainder in original_set:
            original = remainder
            detail = ""
        elif transform_family == "ohe":
            original = _match_one_hot_source(remainder, original_features)
            detail = (
                remainder[len(original) + 1 :]
                if original and remainder.startswith(f"{original}_")
                else ""
            )
        else:
            original = transformed if transformed in original_set else None
            detail = ""

        rows.append(
            {
                "transformed_feature": transformed,
                "original_feature": original,
                "transform_family": transform_family,
                "transformed_detail": detail,
                "mapping_status": "mapped" if original else "unmapped",
            }
        )
    return pd.DataFrame(rows)


def merge_transformed_and_group_mapping(
    transformed_mapping: pd.DataFrame,
    group_mapping: pd.DataFrame,
) -> pd.DataFrame:
    """Attach group labels to transformed-column mappings."""
    return transformed_mapping.merge(
        group_mapping[["original_feature", "feature_group"]],
        on="original_feature",
        how="left",
        validate="many_to_one",
    )

