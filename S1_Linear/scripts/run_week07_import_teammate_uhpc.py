"""
Week 7 - Importing the teammate semantic-recoded 50% UHPC dataset.

This script creates an S1-owned modelling input from the canonical processed
dataset in S2_Kernel. It removes dataset-export artifacts and exact duplicate
rows, but deliberately leaves imputation, encoding, and scaling for train-only
model pipelines.
"""

from pathlib import Path
import argparse
import hashlib
import sys

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

for path in (project_root / "src", project_root, project_root.parent):
    sys.path.insert(0, str(path))

import numpy as np
import pandas as pd

from s1_linear.config import load_config


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a config path relative to S1_Linear."""
    path = Path(path)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def sha256_file(path: Path) -> str:
    """Calculate a file hash for dataset-lineage auditing."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text_values(df: pd.DataFrame) -> pd.DataFrame:
    """Strip text values and convert blank strings to missing values."""
    df = df.copy()
    text_columns = df.select_dtypes(include=["object", "string"]).columns

    for column in text_columns:
        df[column] = df[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
        df[column] = df[column].replace("", np.nan)

    return df


def make_column_audit(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Create a modelling-readiness profile for every output column."""
    rows = []

    for column in df.columns:
        is_numeric = pd.api.types.is_numeric_dtype(df[column])
        missing_count = int(df[column].isna().sum())
        row = {
            "column": column,
            "role": "target" if column == target_col else "predictor",
            "feature_type": "numeric" if is_numeric else "categorical",
            "dtype": str(df[column].dtype),
            "missing_count": missing_count,
            "missing_percentage": missing_count / len(df) * 100 if len(df) else np.nan,
            "unique_values_non_missing": int(df[column].nunique(dropna=True)),
            "constant_including_missing": bool(df[column].nunique(dropna=False) <= 1),
            "zero_count": np.nan,
            "zero_percentage": np.nan,
        }

        if is_numeric:
            zero_count = int(df[column].eq(0).sum())
            row["zero_count"] = zero_count
            row["zero_percentage"] = zero_count / len(df) * 100 if len(df) else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def audit_duplicate_features(
    df: pd.DataFrame,
    target_col: str,
) -> dict[str, int]:
    """Audit repeated feature vectors after exact duplicate rows are removed."""
    feature_columns = [column for column in df.columns if column != target_col]
    duplicate_feature_mask = df.duplicated(subset=feature_columns, keep=False)

    if not duplicate_feature_mask.any():
        return {
            "duplicate_feature_groups": 0,
            "rows_in_duplicate_feature_groups": 0,
            "duplicate_feature_groups_with_conflicting_targets": 0,
        }

    grouped_target_counts = (
        df.loc[duplicate_feature_mask]
        .groupby(feature_columns, dropna=False)[target_col]
        .nunique(dropna=False)
    )

    return {
        "duplicate_feature_groups": int(len(grouped_target_counts)),
        "rows_in_duplicate_feature_groups": int(duplicate_feature_mask.sum()),
        "duplicate_feature_groups_with_conflicting_targets": int(
            grouped_target_counts.gt(1).sum()
        ),
    }


def prepare_teammate_dataset(
    source_df: pd.DataFrame,
    target_col: str,
    cleaning_config: dict,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply only source-level cleaning that is safe before data splitting."""
    df = source_df.copy()
    source_rows, source_columns = df.shape

    required_columns = cleaning_config.get("required_columns", [target_col])
    missing_required = [column for column in required_columns if column not in df]
    if missing_required:
        raise ValueError(f"Required source columns are missing: {missing_required}")

    configured_drop_columns = cleaning_config.get("drop_columns", [])
    present_drop_columns = [
        column for column in configured_drop_columns if column in df
    ]
    missing_drop_columns = [
        column for column in configured_drop_columns if column not in df
    ]
    df = df.drop(columns=present_drop_columns)

    if cleaning_config.get("strip_text_values", True):
        df = normalize_text_values(df)

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    rows_with_missing_target = int(df[target_col].isna().sum())

    if cleaning_config.get("drop_rows_with_missing_target", True):
        df = df.loc[df[target_col].notna()].copy()

    exact_duplicates_before_drop = int(df.duplicated().sum())
    if cleaning_config.get("drop_exact_duplicate_rows", True):
        df = df.drop_duplicates().copy()

    df = df.reset_index(drop=True)
    duplicate_feature_audit = audit_duplicate_features(df, target_col=target_col)

    audit = {
        "source_rows": source_rows,
        "source_columns": source_columns,
        "output_rows": len(df),
        "output_columns": df.shape[1],
        "columns_dropped": present_drop_columns,
        "configured_drop_columns_not_found": missing_drop_columns,
        "rows_with_missing_target_found": rows_with_missing_target,
        "exact_duplicate_rows_removed": exact_duplicates_before_drop,
        "remaining_missing_cells": int(df.isna().sum().sum()),
        "remaining_columns_with_missing_values": int(df.isna().any().sum()),
        "numeric_predictors": int(
            df.drop(columns=target_col).select_dtypes(include=["number"]).shape[1]
        ),
        "categorical_predictors": int(
            df.drop(columns=target_col).select_dtypes(exclude=["number"]).shape[1]
        ),
        **duplicate_feature_audit,
    }

    return df, audit


def make_readiness_report(
    df: pd.DataFrame,
    target_col: str,
    audit: dict[str, object],
) -> pd.DataFrame:
    """Document what is ready now and what belongs in the model pipeline."""
    rows = [
        {
            "check": "target_available",
            "status": "pass" if df[target_col].isna().sum() == 0 else "fail",
            "value": int(df[target_col].isna().sum()),
            "action": "No action needed when value is zero.",
        },
        {
            "check": "source_row_count_vs_expected",
            "status": (
                "pass"
                if audit["source_row_count_difference_from_expected"] == 0
                else "blocked_by_source"
            ),
            "value": (
                f'{audit["source_rows"]} source rows / '
                f'{audit["expected_rows_before_teammate_preprocessing"]} expected'
            ),
            "action": "The import cannot restore rows already removed upstream.",
        },
        {
            "check": "accidental_saved_index_removed",
            "status": "pass" if "Unnamed: 0" not in df.columns else "fail",
            "value": "Unnamed: 0" not in df.columns,
            "action": "Never use a saved row index as a predictor.",
        },
        {
            "check": "redundant_cement_type_removed",
            "status": "pass" if "cement_type" not in df.columns else "review",
            "value": "cement_type" not in df.columns,
            "action": "Use cement_type_clean as the canonical cement category.",
        },
        {
            "check": "exact_duplicate_rows_removed",
            "status": "pass" if df.duplicated().sum() == 0 else "fail",
            "value": audit["exact_duplicate_rows_removed"],
            "action": "Exact feature-and-target duplicates were removed before splitting.",
        },
        {
            "check": "remaining_missing_values",
            "status": "pipeline_required" if df.isna().any().any() else "pass",
            "value": audit["remaining_missing_cells"],
            "action": "Fit imputation and missing indicators on training data only.",
        },
        {
            "check": "categorical_predictors",
            "status": "pipeline_required",
            "value": audit["categorical_predictors"],
            "action": "Fit unknown-safe categorical encoding on training data only.",
        },
        {
            "check": "duplicate_feature_groups",
            "status": "review",
            "value": audit["duplicate_feature_groups"],
            "action": "Use a feature-vector hash as fallback grouping during evaluation.",
        },
        {
            "check": "group_split_identifier",
            "status": (
                "available_separately"
                if audit.get("publication_lineage_available")
                else "source_limitation"
            ),
            "value": audit.get(
                "publication_lineage_path",
                "Mix-ID and paper source are unavailable",
            ),
            "action": (
                "Keep lineage out of predictors; use it for publication-group evaluation."
                if audit.get("publication_lineage_available")
                else "Request preserved grouping columns; feature hashes are only a partial fallback."
            ),
        },
        {
            "check": "preprocessing_fitted_before_split",
            "status": "pass",
            "value": False,
            "action": "This import intentionally performs no imputation, encoding, or scaling.",
        },
    ]

    return pd.DataFrame(rows)


def make_audit_summary(
    source_path: Path,
    output_path: Path,
    audit: dict[str, object],
) -> pd.DataFrame:
    """Create a compact lineage and cleaning summary."""
    rows = [
        {
            "item": "canonical_source_path",
            "value": str(source_path),
            "detail": "Teammate-prepared semantic UHPC dataset.",
        },
        {
            "item": "canonical_source_sha256",
            "value": sha256_file(source_path),
            "detail": "Use this hash to detect upstream dataset changes.",
        },
        {
            "item": "linear_ready_output_path",
            "value": str(output_path),
            "detail": "S1-owned imported modelling input.",
        },
        {
            "item": "linear_ready_output_sha256",
            "value": sha256_file(output_path),
            "detail": "Hash after source-level cleaning.",
        },
    ]

    for item, value in audit.items():
        rows.append(
            {
                "item": item,
                "value": ", ".join(value) if isinstance(value, list) else value,
                "detail": "",
            }
        )

    return pd.DataFrame(rows)


def main(config_path: str) -> None:
    """Import, clean, save, and audit the teammate-prepared UHPC dataset."""
    config = load_config(resolve_project_path(config_path))
    source_path = resolve_project_path(config["data"]["source_path"])
    output_path = resolve_project_path(config["outputs"]["cleaned_dataset_path"])
    tables_dir = resolve_project_path(config["outputs"]["tables_dir"])
    target_col = config["data"]["target"]
    lineage_path = resolve_project_path(config["data"]["lineage_path"])

    if not source_path.exists():
        raise FileNotFoundError(f"Teammate dataset not found: {source_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    source_df = pd.read_csv(source_path)
    cleaned_df, audit = prepare_teammate_dataset(
        source_df=source_df,
        target_col=target_col,
        cleaning_config=config["cleaning"],
    )
    expected_source_rows = config["data"].get(
        "expected_rows_before_teammate_preprocessing",
        audit["source_rows"],
    )
    audit["expected_rows_before_teammate_preprocessing"] = expected_source_rows
    audit["source_row_count_difference_from_expected"] = (
        audit["source_rows"] - expected_source_rows
    )
    audit["policy"] = config["data"]["policy"]
    audit["publication_lineage_available"] = lineage_path.exists()
    audit["publication_lineage_path"] = str(lineage_path)
    cleaned_df.to_csv(output_path, index=False)

    column_audit = make_column_audit(cleaned_df, target_col=target_col)
    readiness_report = make_readiness_report(
        cleaned_df,
        target_col=target_col,
        audit=audit,
    )
    audit_summary = make_audit_summary(
        source_path=source_path,
        output_path=output_path,
        audit=audit,
    )

    column_audit.to_csv(
        tables_dir / config["outputs"]["column_audit_name"],
        index=False,
    )
    readiness_report.to_csv(
        tables_dir / config["outputs"]["readiness_name"],
        index=False,
    )
    audit_summary.to_csv(
        tables_dir / config["outputs"]["audit_summary_name"],
        index=False,
    )

    print("Week 7 teammate dataset import complete.")
    print(f"Canonical source: {source_path}")
    print(f"Linear-ready output: {output_path}")
    print("\nImport audit:")
    print(audit_summary.to_string(index=False))
    print("\nReadiness report:")
    print(readiness_report.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Import the teammate-prepared UHPC dataset into S1_Linear."
    )
    parser.add_argument(
        "--config",
        default="configs/week07_teammate_uhpc_import.yaml",
        help="Config path relative to S1_Linear by default.",
    )
    args = parser.parse_args()
    main(args.config)
