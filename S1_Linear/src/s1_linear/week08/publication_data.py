"""Publication-safe data linkage and audit utilities for Week 8."""

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

from s1_linear.week07_experiments import detect_fiber_group


METADATA_COLUMNS = [
    "semantic_row_id",
    "modeling_row_id",
    "mix_id",
    "publication_country",
    "publication_source",
    "publication_year",
    "publication_reference_id",
    "publication_group",
]


def sha256_file(path: Path) -> str:
    """Return a SHA-256 hash for a data-lineage audit."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text_values(df: pd.DataFrame) -> pd.DataFrame:
    """Strip text values and replace blank strings with missing values."""
    normalized = df.copy()
    for column in normalized.select_dtypes(include=["object", "string"]).columns:
        normalized[column] = normalized[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
        normalized[column] = normalized[column].replace("", np.nan)
    return normalized


def build_aligned_week08_data(
    semantic_df: pd.DataFrame,
    lineage_df: pd.DataFrame,
    target_col: str,
    drop_predictor_columns: list[str],
    week07_linear_ready_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """
    Build predictor and metadata tables with identical row alignment.

    The cleaning sequence deliberately matches the Week 7 import: drop export
    artifacts, normalize text, require a numeric target, and remove exact
    feature-and-target duplicates. Metadata is filtered by the same row mask
    but never enters the predictor table.
    """
    if len(semantic_df) != len(lineage_df):
        raise ValueError("Semantic and lineage row counts do not match.")
    if target_col not in semantic_df:
        raise ValueError(f"Target column is missing: {target_col}")
    if "publication_group" not in lineage_df:
        raise ValueError("Publication lineage lacks publication_group.")

    if "Unnamed: 0" in semantic_df and "semantic_row_id" in lineage_df:
        source_ids = pd.to_numeric(semantic_df["Unnamed: 0"], errors="coerce")
        lineage_ids = pd.to_numeric(lineage_df["semantic_row_id"], errors="coerce")
        if not source_ids.equals(lineage_ids):
            raise ValueError("Semantic row IDs do not align with publication lineage.")

    present_drop_columns = [
        column for column in drop_predictor_columns if column in semantic_df
    ]
    modeling = semantic_df.drop(columns=present_drop_columns).copy()
    modeling = normalize_text_values(modeling)
    modeling[target_col] = pd.to_numeric(modeling[target_col], errors="coerce")

    valid_target_mask = modeling[target_col].notna()
    exact_duplicate_mask = modeling.duplicated(keep="first")
    keep_mask = valid_target_mask & ~exact_duplicate_mask

    modeling = modeling.loc[keep_mask].reset_index(drop=True)
    lineage = lineage_df.loc[keep_mask].reset_index(drop=True).copy()
    lineage.insert(1, "modeling_row_id", np.arange(len(lineage)))
    lineage = lineage[[column for column in METADATA_COLUMNS if column in lineage]]

    metadata_leakage_columns = [
        column for column in METADATA_COLUMNS if column in modeling.columns
    ]
    if metadata_leakage_columns:
        raise ValueError(
            f"Metadata columns leaked into predictors: {metadata_leakage_columns}"
        )
    if lineage["publication_group"].isna().any():
        raise ValueError("Publication groups contain missing values after linkage.")

    matches_week07 = None
    if week07_linear_ready_df is not None:
        matches_week07 = modeling.equals(week07_linear_ready_df.reset_index(drop=True))
        if not matches_week07:
            raise ValueError("Week 8 modeling table does not exactly match Week 7 input.")

    audit = {
        "semantic_source_rows": len(semantic_df),
        "lineage_source_rows": len(lineage_df),
        "rows_with_missing_target_removed": int((~valid_target_mask).sum()),
        "exact_duplicate_rows_removed": int(exact_duplicate_mask.sum()),
        "week08_modeling_rows": len(modeling),
        "week08_modeling_columns_including_target": modeling.shape[1],
        "week08_lineage_rows": len(lineage),
        "publication_groups": int(lineage["publication_group"].nunique()),
        "missing_publication_groups": int(lineage["publication_group"].isna().sum()),
        "metadata_columns_in_predictors": len(metadata_leakage_columns),
        "matches_week07_linear_ready_exactly": matches_week07,
        "semantic_row_ids_unique": bool(lineage["semantic_row_id"].is_unique),
        "modeling_row_ids_unique": bool(lineage["modeling_row_id"].is_unique),
    }
    return modeling, lineage, audit


def make_linkage_audit_table(
    audit: dict[str, object],
    source_paths: dict[str, Path],
) -> pd.DataFrame:
    """Create a readable source-lineage and linkage audit table."""
    rows = []
    for name, path in source_paths.items():
        rows.extend(
            [
                {"item": f"{name}_path", "value": str(path)},
                {"item": f"{name}_sha256", "value": sha256_file(path)},
            ]
        )
    rows.extend({"item": item, "value": value} for item, value in audit.items())
    return pd.DataFrame(rows)


def make_publication_audit(
    modeling_df: pd.DataFrame,
    lineage_df: pd.DataFrame,
    target_col: str,
    important_numeric_features: list[str],
    minimum_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create publication-level composition, target, and missingness audits."""
    if len(modeling_df) != len(lineage_df):
        raise ValueError("Modeling and lineage tables are not row-aligned.")

    features = modeling_df.drop(columns=target_col)
    target = pd.to_numeric(modeling_df[target_col], errors="coerce")
    fiber_group = detect_fiber_group(features).reset_index(drop=True)
    combined = lineage_df.reset_index(drop=True).copy()
    combined[target_col] = target.reset_index(drop=True)
    combined["row_missing_cells"] = features.isna().sum(axis=1).to_numpy()
    combined["fiber_used"] = fiber_group.eq("fiber_used").to_numpy()

    for column in important_numeric_features:
        if column in features:
            combined[column] = pd.to_numeric(features[column], errors="coerce").to_numpy()
    if "curing_method" in features:
        combined["curing_method"] = features["curing_method"].to_numpy()

    audit_rows = []
    for publication_group, group in combined.groupby("publication_group", sort=True):
        row = {
            "publication_group": publication_group,
            "publication_reference_id": group["publication_reference_id"].iloc[0],
            "publication_source": group["publication_source"].iloc[0],
            "publication_country": group["publication_country"].iloc[0],
            "publication_year": group["publication_year"].iloc[0],
            "n_rows": len(group),
            "target_mean": group[target_col].mean(),
            "target_std": group[target_col].std(),
            "target_min": group[target_col].min(),
            "target_max": group[target_col].max(),
            "missing_cells_total": int(group["row_missing_cells"].sum()),
            "missing_cells_per_row_mean": group["row_missing_cells"].mean(),
            "fiber_used_percentage": group["fiber_used"].mean() * 100,
            "eligible_for_reliable_publication_metrics": len(group) >= minimum_rows,
        }
        if "curing_method" in group:
            modes = group["curing_method"].dropna().mode()
            row["dominant_curing_method"] = modes.iloc[0] if not modes.empty else np.nan

        for column in important_numeric_features:
            if column in group:
                row[f"{column}_min"] = group[column].min()
                row[f"{column}_max"] = group[column].max()
        audit_rows.append(row)

    publication_audit = pd.DataFrame(audit_rows).sort_values(
        ["n_rows", "publication_group"],
        ascending=[False, True],
    )
    group_sizes = publication_audit["n_rows"]
    summary = pd.DataFrame(
        [
            {
                "modeling_rows": len(modeling_df),
                "publication_groups": len(publication_audit),
                "minimum_group_rows": int(group_sizes.min()),
                "median_group_rows": float(group_sizes.median()),
                "maximum_group_rows": int(group_sizes.max()),
                "publications_with_at_least_minimum_rows": int(
                    publication_audit[
                        "eligible_for_reliable_publication_metrics"
                    ].sum()
                ),
                "rows_in_publications_with_at_least_minimum_rows": int(
                    publication_audit.loc[
                        publication_audit[
                            "eligible_for_reliable_publication_metrics"
                        ],
                        "n_rows",
                    ].sum()
                ),
                "largest_publication_row_percentage": float(
                    group_sizes.max() / len(modeling_df) * 100
                ),
            }
        ]
    )
    return publication_audit, summary
