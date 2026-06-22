"""
Week 6 - UHPC semantic missingness dataset builder.

This script starts from the Week 5 cleaned UHPC file, keeps rows with a valid
28-day compressive-strength target, excludes leakage columns, applies semantic
missingness rules, and writes train/validation/test splits for each feature
policy.
"""

from pathlib import Path
import argparse
import sys

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

for path in (project_root / "src", project_root, project_root.parent):
    sys.path.insert(0, str(path))

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from s1_linear.config import load_config
from s1_linear.feature_policies import (
    TARGET_COLUMN,
    build_week6_preprocessor,
    coerce_numeric_features,
    filter_features_by_missingness,
    make_excluded_columns_table,
    select_week6_candidate_features,
)
from s1_linear.semantic_missingness import (
    make_missing_before_after_report,
    normalize_text_missing_values,
    recode_semantic_missingness,
)


def resolve_project_path(path: str | Path) -> Path:
    """Resolve config paths relative to S1_Linear/."""
    path = Path(path)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return project_root / path


def ensure_dirs(*paths: Path) -> None:
    """Create output folders."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def load_week6_source(config: dict) -> pd.DataFrame:
    """Load the Week 5 CSV or Excel file configured for Week 6."""
    data_config = config["data"]
    source_path = resolve_project_path(data_config["source_path"])

    if not source_path.exists():
        raise FileNotFoundError(f"Week 6 source data not found: {source_path}")

    if source_path.suffix.lower() in [".xls", ".xlsx"]:
        sheet_name = data_config.get("sheet_name", 0)
        return pd.read_excel(source_path, sheet_name=sheet_name)

    return pd.read_csv(source_path)


def keep_valid_target_rows(
    df: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only rows with a numeric 28-day target value."""
    if target_col not in df.columns:
        raise ValueError(
            f"Target column not found: {target_col}\n"
            f"Available columns: {list(df.columns)}"
        )

    total_rows = len(df)
    target = pd.to_numeric(df[target_col], errors="coerce")
    valid_target = target.notna()

    filtered = df.loc[valid_target].copy()
    filtered[target_col] = target.loc[valid_target]

    summary = pd.DataFrame(
        [
            {"item": "total_rows_in_source", "value": total_rows},
            {"item": "rows_with_valid_28_day_target", "value": int(valid_target.sum())},
            {
                "item": "rows_missing_28_day_target",
                "value": int((~valid_target).sum()),
            },
            {
                "item": "usable_rows_for_28_day_target_percent",
                "value": valid_target.mean() * 100,
            },
        ]
    )

    return filtered, summary


def split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    val_size: float,
    test_size: float,
    random_state: int,
):
    """Create train/validation/test splits without fitting preprocessing."""
    temp_size = val_size + test_size
    if temp_size <= 0 or temp_size >= 1:
        raise ValueError("val_size + test_size must be greater than 0 and less than 1.")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=temp_size,
        random_state=random_state,
    )

    relative_test_size = test_size / temp_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test_size,
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def save_split_files(
    policy_dir: Path,
    target_col: str,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
) -> None:
    """Save Week 6 split files in a policy-specific folder."""
    ensure_dirs(policy_dir)

    X_train.to_csv(policy_dir / "X_train.csv", index=False)
    X_val.to_csv(policy_dir / "X_val.csv", index=False)
    X_test.to_csv(policy_dir / "X_test.csv", index=False)

    y_train.to_frame(name=target_col).to_csv(policy_dir / "y_train.csv", index=False)
    y_val.to_frame(name=target_col).to_csv(policy_dir / "y_val.csv", index=False)
    y_test.to_frame(name=target_col).to_csv(policy_dir / "y_test.csv", index=False)


def build_preprocessing_variants(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessing_config: dict,
    random_state: int,
) -> dict[str, object]:
    """Build train-profiled preprocessors for the configured preprocessing policies."""
    common_kwargs = {
        "numeric_imputer": preprocessing_config.get("numeric_imputer", "simple"),
        "numeric_add_indicator": preprocessing_config.get(
            "numeric_add_indicator", True
        ),
        "knn_neighbors": preprocessing_config.get("knn_neighbors", 5),
        "low_cardinality_max": preprocessing_config.get("low_cardinality_max", 10),
        "medium_cardinality_max": preprocessing_config.get(
            "medium_cardinality_max", 20
        ),
        "random_state": random_state,
    }

    builds = {
        "grouped_ohe": build_week6_preprocessor(
            X_train=X_train,
            y_train=y_train,
            high_cardinality_policy="grouped_ohe",
            **common_kwargs,
        )
    }

    if preprocessing_config.get(
        "enable_target_encoder_high_cardinality",
        preprocessing_config.get("target_encoder_high_cardinality", False),
    ):
        builds["target_encoder"] = build_week6_preprocessor(
            X_train=X_train,
            y_train=y_train,
            high_cardinality_policy="target_encoder",
            **common_kwargs,
        )

    return builds


def add_policy_columns(
    df: pd.DataFrame,
    policy_name: str,
    preprocessing_policy: str | None = None,
) -> pd.DataFrame:
    """Attach policy labels to an audit/report table without mutating it."""
    df = df.copy()
    df.insert(0, "policy", policy_name)
    if preprocessing_policy is not None:
        if "preprocessing_policy" in df.columns:
            df["preprocessing_policy"] = preprocessing_policy
        else:
            df.insert(1, "preprocessing_policy", preprocessing_policy)
    return df


def make_encoding_expansion_audit(
    policy_name: str,
    preprocessing_policy: str,
    X_train: pd.DataFrame,
    build,
    fitted_preprocessor,
) -> pd.DataFrame:
    """Summarize train-fitted encoding and missing-indicator column expansion."""

    def output_width(indexer) -> int:
        if isinstance(indexer, slice):
            return max(0, (indexer.stop or 0) - (indexer.start or 0))
        return len(indexer)

    output_indices = fitted_preprocessor.output_indices_
    numeric_output_columns = output_width(
        output_indices.get("numeric", slice(0, 0))
    )
    encoded_categorical_columns = sum(
        output_width(indexer)
        for name, indexer in output_indices.items()
        if name.startswith("categorical_")
    )
    transformed_columns = len(fitted_preprocessor.get_feature_names_out())
    numeric_source_columns = len(build.numeric_features)
    categorical_source_columns = len(build.categorical_features)

    return pd.DataFrame(
        [
            {
                "policy": policy_name,
                "preprocessing_policy": preprocessing_policy,
                "fit_scope": "X_train_only",
                "input_predictor_columns": X_train.shape[1],
                "numeric_source_columns": numeric_source_columns,
                "categorical_source_columns": categorical_source_columns,
                "encoded_categorical_columns": encoded_categorical_columns,
                "extra_columns_from_categorical_encoding": (
                    encoded_categorical_columns - categorical_source_columns
                ),
                "extra_numeric_missing_indicator_columns": (
                    numeric_output_columns - numeric_source_columns
                ),
                "total_transformed_columns": transformed_columns,
                "net_extra_columns_after_preprocessing": (
                    transformed_columns - X_train.shape[1]
                ),
            }
        ]
    )


def run_policy(
    policy_name: str,
    policy_config: dict,
    base_X: pd.DataFrame,
    y: pd.Series,
    target_col: str,
    split_config: dict,
    processed_dir: Path,
    preprocessor_dir: Path,
    fit_preprocessors: bool,
    preprocessing_config: dict,
) -> tuple[
    dict,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Build one feature-policy dataset and return its audit tables."""
    missing_threshold = policy_config.get("missing_threshold")
    apply_semantic_recode = policy_config.get("apply_semantic_recode", True)

    selected_features, feature_table = filter_features_by_missingness(
        df=base_X,
        features=list(base_X.columns),
        missing_threshold=missing_threshold,
    )

    X_before_recode = base_X[selected_features].copy()

    if apply_semantic_recode:
        X_final, semantic_summary = recode_semantic_missingness(X_before_recode)
    else:
        X_final = X_before_recode.copy()
        semantic_summary = pd.DataFrame()

    missing_report = make_missing_before_after_report(X_before_recode, X_final)

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        X=X_final,
        y=y,
        val_size=split_config["val_size"],
        test_size=split_config["test_size"],
        random_state=split_config["random_state"],
    )

    preprocessing_builds = build_preprocessing_variants(
        X_train=X_train,
        y_train=y_train,
        preprocessing_config=preprocessing_config,
        random_state=split_config["random_state"],
    )
    primary_build = preprocessing_builds["grouped_ohe"]
    dropped_numeric_features = primary_build.dropped_numeric_features

    if dropped_numeric_features:
        X_train = X_train.drop(columns=dropped_numeric_features)
        X_val = X_val.drop(columns=dropped_numeric_features)
        X_test = X_test.drop(columns=dropped_numeric_features)

    numeric_audit = add_policy_columns(
        primary_build.numeric_missing_audit,
        policy_name=policy_name,
    )
    cardinality_report = add_policy_columns(
        primary_build.cardinality_report,
        policy_name=policy_name,
    )
    preprocessing_summary = pd.concat(
        [
            add_policy_columns(
                build.preprocessing_summary,
                policy_name=policy_name,
                preprocessing_policy=preprocessing_policy,
            )
            for preprocessing_policy, build in preprocessing_builds.items()
        ],
        ignore_index=True,
    )

    policy_dir = processed_dir / policy_name
    save_split_files(
        policy_dir=policy_dir,
        target_col=target_col,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    pd.DataFrame({"feature": selected_features}).to_csv(
        policy_dir / "selected_features.csv",
        index=False,
    )
    pd.DataFrame({"feature": list(X_train.columns)}).to_csv(
        policy_dir / "modeling_features.csv",
        index=False,
    )
    feature_table.to_csv(policy_dir / "feature_missingness_policy.csv", index=False)
    missing_report.to_csv(policy_dir / "missing_before_after_recode.csv", index=False)
    numeric_audit.to_csv(policy_dir / "numeric_fully_missing_audit.csv", index=False)
    cardinality_report.to_csv(
        policy_dir / "categorical_cardinality_report.csv",
        index=False,
    )
    preprocessing_summary.to_csv(
        policy_dir / "preprocessing_summary.csv",
        index=False,
    )

    if not semantic_summary.empty:
        semantic_summary.insert(0, "policy", policy_name)
        semantic_summary.to_csv(policy_dir / "semantic_recode_summary.csv", index=False)

    encoding_expansion_frames = []
    if fit_preprocessors:
        ensure_dirs(preprocessor_dir)
        for preprocessing_policy, build in preprocessing_builds.items():
            preprocessor = build.preprocessor
            preprocessor.fit(X_train, y_train)
            encoding_expansion_frames.append(
                make_encoding_expansion_audit(
                    policy_name=policy_name,
                    preprocessing_policy=preprocessing_policy,
                    X_train=X_train,
                    build=build,
                    fitted_preprocessor=preprocessor,
                )
            )

            if preprocessing_policy == "grouped_ohe":
                filename = f"week6_{policy_name}_preprocessor.joblib"
            else:
                filename = (
                    f"week6_{policy_name}_{preprocessing_policy}_preprocessor.joblib"
                )

            joblib.dump(preprocessor, preprocessor_dir / filename)

    encoding_expansion = (
        pd.concat(encoding_expansion_frames, ignore_index=True)
        if encoding_expansion_frames
        else pd.DataFrame()
    )
    if not encoding_expansion.empty:
        encoding_expansion.to_csv(
            policy_dir / "encoding_column_expansion.csv",
            index=False,
        )

    policy_summary = {
        "policy": policy_name,
        "missing_threshold_before_semantic_recode": missing_threshold,
        "semantic_recode_applied": apply_semantic_recode,
        "rows_total": len(X_final),
        "features_before_policy": base_X.shape[1],
        "features_after_policy": len(selected_features),
        "features_dropped_by_policy": base_X.shape[1] - len(selected_features),
        "numeric_features_detected": len(primary_build.numeric_missing_audit),
        "numeric_features_dropped_fully_missing_train": len(
            dropped_numeric_features
        ),
        "numeric_features": len(primary_build.numeric_features),
        "categorical_features": len(primary_build.categorical_features),
        "low_cardinality_categorical_features": len(
            primary_build.low_cardinality_features
        ),
        "medium_cardinality_categorical_features": len(
            primary_build.medium_cardinality_features
        ),
        "high_cardinality_categorical_features": len(
            primary_build.high_cardinality_features
        ),
        "train_rows": len(X_train),
        "validation_rows": len(X_val),
        "test_rows": len(X_test),
    }

    feature_table.insert(0, "policy", policy_name)
    missing_report.insert(0, "policy", policy_name)

    return (
        policy_summary,
        feature_table,
        missing_report,
        semantic_summary,
        numeric_audit,
        cardinality_report,
        preprocessing_summary,
        encoding_expansion,
    )


def main(config_path: str) -> None:
    """Run the full Week 6 dataset preparation workflow."""
    config = load_config(resolve_project_path(config_path))

    target_col = config["data"].get("target", TARGET_COLUMN)
    preprocessing_config = config.get("preprocessing", {})
    processed_dir = resolve_project_path(config["outputs"]["processed_dir"])
    tables_dir = resolve_project_path(config["outputs"]["tables_dir"])
    preprocessor_dir = resolve_project_path(config["outputs"]["preprocessor_dir"])
    semantic_cleaned_path = resolve_project_path(
        config["outputs"]["semantic_cleaned_path"]
    )

    ensure_dirs(
        processed_dir, tables_dir, preprocessor_dir, semantic_cleaned_path.parent
    )

    raw_df = load_week6_source(config)
    raw_df = normalize_text_missing_values(raw_df)

    target_df, target_summary = keep_valid_target_rows(raw_df, target_col)
    candidate_features = select_week6_candidate_features(target_df, target_col)

    if not candidate_features:
        raise ValueError("No Week 6 candidate features were found.")

    excluded_columns = make_excluded_columns_table(
        df=target_df,
        selected_features=candidate_features,
        target_col=target_col,
    )

    base_X = target_df[candidate_features].copy()
    base_X, numeric_coercion_report = coerce_numeric_features(base_X)
    y = target_df[target_col].copy()

    full_semantic_X, full_semantic_summary = recode_semantic_missingness(base_X)
    full_missing_report = make_missing_before_after_report(base_X, full_semantic_X)

    semantic_cleaned_df = pd.concat(
        [full_semantic_X, y.rename(target_col)],
        axis=1,
    )
    semantic_cleaned_df.to_csv(semantic_cleaned_path, index=False)

    target_summary.to_csv(tables_dir / "week06_target_row_count.csv", index=False)
    excluded_columns.to_csv(tables_dir / "week06_excluded_columns.csv", index=False)
    numeric_coercion_report.to_csv(
        tables_dir / "week06_numeric_coercion_report.csv",
        index=False,
    )
    full_semantic_summary.to_csv(
        tables_dir / "week06_full_semantic_recode_summary.csv",
        index=False,
    )
    full_missing_report.to_csv(
        tables_dir / "week06_full_missing_before_after_recode.csv",
        index=False,
    )

    policy_summaries = []
    all_feature_tables = []
    all_missing_reports = []
    all_semantic_summaries = []
    all_numeric_audits = []
    all_cardinality_reports = []
    all_preprocessing_summaries = []
    all_encoding_expansion_reports = []

    for policy_name, policy_config in config["policies"].items():
        print(f"\nBuilding Week 6 policy: {policy_name}")
        (
            policy_summary,
            feature_table,
            missing_report,
            semantic_summary,
            numeric_audit,
            cardinality_report,
            preprocessing_summary,
            encoding_expansion,
        ) = run_policy(
            policy_name=policy_name,
            policy_config=policy_config,
            base_X=base_X,
            y=y,
            target_col=target_col,
            split_config=config["split"],
            processed_dir=processed_dir,
            preprocessor_dir=preprocessor_dir,
            fit_preprocessors=config["outputs"].get("fit_preprocessors", True),
            preprocessing_config=preprocessing_config,
        )

        policy_summaries.append(policy_summary)
        all_feature_tables.append(feature_table)
        all_missing_reports.append(missing_report)
        all_numeric_audits.append(numeric_audit)
        all_cardinality_reports.append(cardinality_report)
        all_preprocessing_summaries.append(preprocessing_summary)
        if not encoding_expansion.empty:
            all_encoding_expansion_reports.append(encoding_expansion)

        if not semantic_summary.empty:
            all_semantic_summaries.append(semantic_summary)

    policy_summary_df = pd.DataFrame(policy_summaries)
    policy_summary_df.to_csv(
        tables_dir / "week06_feature_policy_summary.csv", index=False
    )
    policy_summary_df[
        ["policy", "rows_total", "train_rows", "validation_rows", "test_rows"]
    ].to_csv(tables_dir / "week06_split_sizes.csv", index=False)

    pd.concat(all_feature_tables, ignore_index=True).to_csv(
        tables_dir / "week06_feature_policy_missingness.csv",
        index=False,
    )
    pd.concat(all_missing_reports, ignore_index=True).to_csv(
        tables_dir / "week06_policy_missing_before_after_recode.csv",
        index=False,
    )
    pd.concat(all_numeric_audits, ignore_index=True).to_csv(
        tables_dir / "week06_numeric_fully_missing_audit.csv",
        index=False,
    )
    pd.concat(all_cardinality_reports, ignore_index=True).to_csv(
        tables_dir / "week06_categorical_cardinality_report.csv",
        index=False,
    )
    pd.concat(all_preprocessing_summaries, ignore_index=True).to_csv(
        tables_dir / "week06_preprocessing_summary.csv",
        index=False,
    )

    encoding_expansion_df = (
        pd.concat(all_encoding_expansion_reports, ignore_index=True)
        if all_encoding_expansion_reports
        else pd.DataFrame()
    )
    if not encoding_expansion_df.empty:
        encoding_expansion_df.to_csv(
            tables_dir / "week06_encoding_column_expansion.csv",
            index=False,
        )

    if all_semantic_summaries:
        pd.concat(all_semantic_summaries, ignore_index=True).to_csv(
            tables_dir / "week06_policy_semantic_recode_summary.csv",
            index=False,
        )

    print("\nWeek 6 dataset preparation complete.")
    print(target_summary.to_string(index=False))
    print("\nFeature policy summary:")
    print(policy_summary_df.to_string(index=False))
    if not encoding_expansion_df.empty:
        print("\nEncoded-column expansion (preprocessing fitted on X_train only):")
        for row in encoding_expansion_df.itertuples(index=False):
            print(
                f"- {row.policy} / {row.preprocessing_policy}: categorical "
                f"{row.categorical_source_columns} -> "
                f"{row.encoded_categorical_columns} "
                f"(+{row.extra_columns_from_categorical_encoding}); "
                f"numeric missing indicators +"
                f"{row.extra_numeric_missing_indicator_columns}; total "
                f"{row.input_predictor_columns} -> {row.total_transformed_columns} "
                f"(+{row.net_extra_columns_after_preprocessing})."
            )
    else:
        print("\nEncoded-column expansion not calculated: preprocessor fitting is disabled.")
    print(f"\nSaved semantic cleaned file to: {semantic_cleaned_path}")
    print(f"Saved policy split folders to: {processed_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare Week 6 UHPC semantic-missingness datasets."
    )
    parser.add_argument(
        "--config",
        default="configs/week06_semantic_missingness.yaml",
        help="Path to the Week 6 YAML config, relative to S1_Linear/ by default.",
    )
    args = parser.parse_args()
    main(args.config)
