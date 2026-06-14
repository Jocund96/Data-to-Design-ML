"""Train-only preprocessing utilities for the teammate UHPC 50% dataset."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class Week07PreprocessorBuild:
    """Preprocessor and train-derived feature reports."""

    preprocessor: ColumnTransformer
    numeric_features: list[str]
    categorical_features: list[str]
    train_missingness_report: pd.DataFrame
    categorical_cardinality_report: pd.DataFrame


def make_feature_hash_groups(X: pd.DataFrame) -> pd.Series:
    """Create stable fallback groups so identical feature vectors stay together."""
    normalized = X.copy()
    text_columns = normalized.select_dtypes(exclude=["number"]).columns

    for column in text_columns:
        normalized[column] = normalized[column].fillna("__MISSING__").astype(str)

    groups = pd.util.hash_pandas_object(normalized, index=False)
    groups.name = "feature_hash_group"
    return groups


def make_train_missingness_report(X_train: pd.DataFrame) -> pd.DataFrame:
    """Profile missingness using the training split only."""
    rows = []

    for column in X_train.columns:
        missing_count = int(X_train[column].isna().sum())
        rows.append(
            {
                "column": column,
                "feature_type": (
                    "numeric"
                    if pd.api.types.is_numeric_dtype(X_train[column])
                    else "categorical"
                ),
                "train_rows": len(X_train),
                "missing_count_train": missing_count,
                "missing_percentage_train": (
                    missing_count / len(X_train) * 100 if len(X_train) else np.nan
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["missing_percentage_train", "column"],
        ascending=[False, True],
    )


def make_categorical_cardinality_report(
    X_train: pd.DataFrame,
    categorical_features: list[str],
    missing_value: str,
    min_frequency: int,
    max_categories: int,
) -> pd.DataFrame:
    """Profile categorical cardinality from the training split only."""
    rows = []

    for column in categorical_features:
        filled = X_train[column].fillna(missing_value).astype(str)
        value_counts = filled.value_counts(dropna=False)
        rare_categories = int((value_counts < min_frequency).sum())

        rows.append(
            {
                "column": column,
                "missing_count_train": int(X_train[column].isna().sum()),
                "unique_categories_train": int(filled.nunique(dropna=False)),
                "categories_below_min_frequency": rare_categories,
                "min_frequency": min_frequency,
                "max_categories": max_categories,
                "encoder": "OneHotEncoder(handle_unknown='infrequent_if_exist')",
            }
        )

    return pd.DataFrame(rows)


def build_week07_preprocessor(
    X_train: pd.DataFrame,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    numeric_add_indicator: bool = True,
    categorical_missing_value: str = "missing_reported_gap",
    categorical_min_frequency: int = 5,
    categorical_max_categories: int = 25,
) -> Week07PreprocessorBuild:
    """Build an unfitted preprocessor using feature types found in X_train."""
    if numeric_features is None:
        numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
    else:
        numeric_features = [column for column in numeric_features if column in X_train]

    if categorical_features is None:
        categorical_features = X_train.select_dtypes(exclude=["number"]).columns.tolist()
    else:
        categorical_features = [
            column for column in categorical_features if column in X_train
        ]

    if not numeric_features:
        raise ValueError("No numeric predictors were found in the training split.")

    transformers = [
        (
            "numeric",
            Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median",
                            add_indicator=numeric_add_indicator,
                        ),
                    ),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_features,
        )
    ]

    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant",
                                fill_value=categorical_missing_value,
                            ),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="infrequent_if_exist",
                                min_frequency=categorical_min_frequency,
                                max_categories=categorical_max_categories,
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return Week07PreprocessorBuild(
        preprocessor=preprocessor,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        train_missingness_report=make_train_missingness_report(X_train),
        categorical_cardinality_report=make_categorical_cardinality_report(
            X_train=X_train,
            categorical_features=categorical_features,
            missing_value=categorical_missing_value,
            min_frequency=categorical_min_frequency,
            max_categories=categorical_max_categories,
        ),
    )
