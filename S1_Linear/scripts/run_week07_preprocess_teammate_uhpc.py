"""
Week 7 - Split, scale, and encode the teammate semantic-recoded 50% UHPC data.

The preprocessor is fitted on X_train only. Validation and test data only call
transform, preventing imputation, scaling, and category-learning leakage.
"""

from pathlib import Path
import argparse
import sys

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

for path in (project_root / "src", project_root, project_root.parent):
    sys.path.insert(0, str(path))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from s1_linear.config import load_config
from s1_linear.week07_preprocessing import (
    build_week07_preprocessor,
    make_feature_hash_groups,
)


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a config path relative to S1_Linear."""
    path = Path(path)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def split_feature_hash_grouped(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: float,
    validation_size: float,
    test_size: float,
    random_state: int,
):
    """Create 70/15/15-style splits while keeping identical features together."""
    if not np.isclose(train_size + validation_size + test_size, 1.0):
        raise ValueError("train_size + validation_size + test_size must equal 1.")

    groups = make_feature_hash_groups(X)
    first_split = GroupShuffleSplit(
        n_splits=1,
        train_size=train_size,
        random_state=random_state,
    )
    train_positions, temp_positions = next(first_split.split(X, y, groups=groups))

    X_train = X.iloc[train_positions].copy()
    y_train = y.iloc[train_positions].copy()
    X_temp = X.iloc[temp_positions].copy()
    y_temp = y.iloc[temp_positions].copy()
    temp_groups = groups.iloc[temp_positions]

    relative_validation_size = validation_size / (validation_size + test_size)
    second_split = GroupShuffleSplit(
        n_splits=1,
        train_size=relative_validation_size,
        random_state=random_state,
    )
    val_positions, test_positions = next(
        second_split.split(X_temp, y_temp, groups=temp_groups)
    )

    X_val = X_temp.iloc[val_positions].copy()
    y_val = y_temp.iloc[val_positions].copy()
    X_test = X_temp.iloc[test_positions].copy()
    y_test = y_temp.iloc[test_positions].copy()

    return X_train, X_val, X_test, y_train, y_val, y_test


def save_raw_splits(
    split_dir: Path,
    target_col: str,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
) -> None:
    """Save untransformed splits for pipeline-based model training."""
    split_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(split_dir / "X_train.csv", index=False)
    X_val.to_csv(split_dir / "X_val.csv", index=False)
    X_test.to_csv(split_dir / "X_test.csv", index=False)
    y_train.to_frame(target_col).to_csv(split_dir / "y_train.csv", index=False)
    y_val.to_frame(target_col).to_csv(split_dir / "y_val.csv", index=False)
    y_test.to_frame(target_col).to_csv(split_dir / "y_test.csv", index=False)


def save_transformed_splits(
    transformed_dir: Path,
    feature_names: list[str],
    Xt_train,
    Xt_val,
    Xt_test,
) -> None:
    """Save transformed features for inspection and non-pipeline consumers."""
    transformed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(Xt_train, columns=feature_names).to_csv(
        transformed_dir / "X_train_transformed.csv",
        index=False,
    )
    pd.DataFrame(Xt_val, columns=feature_names).to_csv(
        transformed_dir / "X_val_transformed.csv",
        index=False,
    )
    pd.DataFrame(Xt_test, columns=feature_names).to_csv(
        transformed_dir / "X_test_transformed.csv",
        index=False,
    )


def count_crossing_groups(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> int:
    """Count identical-feature groups that occur in more than one split."""
    rows = []
    for split_name, X_split in [
        ("train", X_train),
        ("validation", X_val),
        ("test", X_test),
    ]:
        rows.append(
            pd.DataFrame(
                {
                    "group": make_feature_hash_groups(X_split).to_numpy(),
                    "split": split_name,
                }
            )
        )

    group_splits = pd.concat(rows, ignore_index=True).groupby("group")["split"].nunique()
    return int(group_splits.gt(1).sum())


def main(config_path: str) -> None:
    """Run Week 7 grouped splitting and train-only preprocessing."""
    config = load_config(resolve_project_path(config_path))
    input_path = resolve_project_path(config["data"]["input_path"])
    target_col = config["data"]["target"]
    split_dir = resolve_project_path(config["outputs"]["split_dir"])
    transformed_dir = resolve_project_path(config["outputs"]["transformed_dir"])
    preprocessor_path = resolve_project_path(config["outputs"]["preprocessor_path"])
    tables_dir = resolve_project_path(config["outputs"]["tables_dir"])

    if not input_path.exists():
        raise FileNotFoundError(
            f"Imported teammate dataset not found: {input_path}\n"
            "Run scripts/run_week07_import_teammate_uhpc.py first."
        )

    df = pd.read_csv(input_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column not found: {target_col}")

    X = df.drop(columns=target_col)
    y = pd.to_numeric(df[target_col], errors="raise")
    split_config = config["split"]

    X_train, X_val, X_test, y_train, y_val, y_test = split_feature_hash_grouped(
        X=X,
        y=y,
        train_size=split_config["train_size"],
        validation_size=split_config["validation_size"],
        test_size=split_config["test_size"],
        random_state=split_config["random_state"],
    )

    preprocessing_config = config["preprocessing"]
    build = build_week07_preprocessor(
        X_train=X_train,
        numeric_add_indicator=preprocessing_config["numeric_add_indicator"],
        categorical_missing_value=preprocessing_config[
            "categorical_missing_value"
        ],
        categorical_min_frequency=preprocessing_config[
            "categorical_min_frequency"
        ],
        categorical_max_categories=preprocessing_config[
            "categorical_max_categories"
        ],
    )

    preprocessor = build.preprocessor
    Xt_train = preprocessor.fit_transform(X_train)
    Xt_val = preprocessor.transform(X_val)
    Xt_test = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out().tolist()

    split_dir.mkdir(parents=True, exist_ok=True)
    transformed_dir.mkdir(parents=True, exist_ok=True)
    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    save_raw_splits(
        split_dir=split_dir,
        target_col=target_col,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )
    save_transformed_splits(
        transformed_dir=transformed_dir,
        feature_names=feature_names,
        Xt_train=Xt_train,
        Xt_val=Xt_val,
        Xt_test=Xt_test,
    )
    joblib.dump(preprocessor, preprocessor_path)

    crossing_groups = count_crossing_groups(X_train, X_val, X_test)
    split_summary = pd.DataFrame(
        [
            {
                "policy": config["data"]["policy"],
                "split_strategy": split_config["strategy"],
                "total_rows": len(df),
                "train_rows": len(X_train),
                "validation_rows": len(X_val),
                "test_rows": len(X_test),
                "train_percentage": len(X_train) / len(df) * 100,
                "validation_percentage": len(X_val) / len(df) * 100,
                "test_percentage": len(X_test) / len(df) * 100,
                "train_target_mean": y_train.mean(),
                "validation_target_mean": y_val.mean(),
                "test_target_mean": y_test.mean(),
                "train_target_std": y_train.std(),
                "validation_target_std": y_val.std(),
                "test_target_std": y_test.std(),
                "feature_hash_groups_crossing_splits": crossing_groups,
                "random_state": split_config["random_state"],
            }
        ]
    )
    preprocessing_summary = pd.DataFrame(
        [
            {
                "policy": config["data"]["policy"],
                "raw_predictors": X.shape[1],
                "numeric_predictors": len(build.numeric_features),
                "categorical_predictors": len(build.categorical_features),
                "transformed_predictors": len(feature_names),
                "numeric_imputer": preprocessing_config["numeric_imputer"],
                "numeric_missing_indicators": preprocessing_config[
                    "numeric_add_indicator"
                ],
                "numeric_scaler": "StandardScaler",
                "categorical_missing_value": preprocessing_config[
                    "categorical_missing_value"
                ],
                "categorical_encoder": (
                    "OneHotEncoder(handle_unknown='infrequent_if_exist', "
                    f"min_frequency={preprocessing_config['categorical_min_frequency']}, "
                    f"max_categories={preprocessing_config['categorical_max_categories']})"
                ),
                "preprocessor_fit_rows": len(X_train),
                "remaining_nan_train": int(np.isnan(Xt_train).sum()),
                "remaining_nan_validation": int(np.isnan(Xt_val).sum()),
                "remaining_nan_test": int(np.isnan(Xt_test).sum()),
            }
        ]
    )

    split_summary.to_csv(
        tables_dir / config["outputs"]["split_summary_name"],
        index=False,
    )
    preprocessing_summary.to_csv(
        tables_dir / config["outputs"]["preprocessing_summary_name"],
        index=False,
    )
    pd.DataFrame({"transformed_feature": feature_names}).to_csv(
        tables_dir / config["outputs"]["feature_names_name"],
        index=False,
    )
    build.categorical_cardinality_report.to_csv(
        tables_dir / config["outputs"]["categorical_report_name"],
        index=False,
    )
    build.train_missingness_report.to_csv(
        tables_dir / config["outputs"]["missingness_report_name"],
        index=False,
    )

    print("Week 7 teammate 50% preprocessing complete.")
    print("\nSplit summary:")
    print(split_summary.to_string(index=False))
    print("\nPreprocessing summary:")
    print(preprocessing_summary.to_string(index=False))
    print(f"\nSaved fitted preprocessor to: {preprocessor_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare the teammate semantic-recoded 50% UHPC dataset."
    )
    parser.add_argument(
        "--config",
        default="configs/week07_teammate_uhpc_preprocessing.yaml",
        help="Config path relative to S1_Linear by default.",
    )
    args = parser.parse_args()
    main(args.config)
