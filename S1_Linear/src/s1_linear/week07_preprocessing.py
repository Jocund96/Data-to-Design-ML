"""Week 7 preprocessing wrappers for the shared UHPC strategy."""

from dataclasses import dataclass

import pandas as pd

from s1_linear.shared_strategies.uhpc_semantic_50 import (
    SHARED_NUMERIC_FEATURES,
    SHARED_ONE_HOT_FEATURES,
    SHARED_TARGET_ENCODED_FEATURES,
    build_shared_uhpc_preprocessor,
    make_categorical_cardinality_report,
    make_train_missingness_report,
)


@dataclass(frozen=True)
class Week07PreprocessorBuild:
    """Preprocessor and train-derived feature reports."""

    preprocessor: object
    numeric_features: list[str]
    categorical_features: list[str]
    one_hot_features: list[str]
    target_encoded_features: list[str]
    train_missingness_report: pd.DataFrame
    categorical_cardinality_report: pd.DataFrame
    column_contract_report: pd.DataFrame


def make_feature_hash_groups(X: pd.DataFrame) -> pd.Series:
    """Create stable fallback groups so identical feature vectors stay together."""
    normalized = X.copy()
    text_columns = normalized.select_dtypes(exclude=["number"]).columns

    for column in text_columns:
        normalized[column] = normalized[column].fillna("__MISSING__").astype(str)

    groups = pd.util.hash_pandas_object(normalized, index=False)
    groups.name = "feature_hash_group"
    return groups


def build_week07_preprocessor(
    X_train: pd.DataFrame,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    numeric_add_indicator: bool = False,
    categorical_missing_value: str = "unused_shared_strategy",
    categorical_min_frequency: int = 0,
    categorical_max_categories: int = 0,
) -> Week07PreprocessorBuild:
    """
    Build an unfitted shared UHPC preprocessor.

    The extra keyword arguments are accepted for backward compatibility with
    older Week 7 configs. The shared strategy intentionally does not add
    numeric missing indicators or rare-category grouping because those choices
    would change the agreed transformed column count.
    """
    del numeric_add_indicator
    del categorical_missing_value
    del categorical_min_frequency
    del categorical_max_categories

    if numeric_features is None:
        numeric_features = [
            column for column in SHARED_NUMERIC_FEATURES if column in X_train
        ]
    if categorical_features is None:
        one_hot_features = [
            column for column in SHARED_ONE_HOT_FEATURES if column in X_train
        ]
        target_encoded_features = [
            column for column in SHARED_TARGET_ENCODED_FEATURES if column in X_train
        ]
    else:
        categorical_set = set(categorical_features)
        one_hot_features = [
            column
            for column in SHARED_ONE_HOT_FEATURES
            if column in X_train and column in categorical_set
        ]
        target_encoded_features = [
            column
            for column in SHARED_TARGET_ENCODED_FEATURES
            if column in X_train and column in categorical_set
        ]
        assigned = set(one_hot_features) | set(target_encoded_features)
        extra_categoricals = [
            column
            for column in categorical_features
            if column in X_train and column not in assigned
        ]
        for column in extra_categoricals:
            if X_train[column].nunique(dropna=True) <= 10:
                one_hot_features.append(column)
            else:
                target_encoded_features.append(column)

    shared = build_shared_uhpc_preprocessor(
        X_train=X_train,
        numeric_features=numeric_features,
        one_hot_features=one_hot_features,
        target_encoded_features=target_encoded_features,
    )
    return Week07PreprocessorBuild(
        preprocessor=shared.preprocessor,
        numeric_features=shared.numeric_features,
        categorical_features=shared.categorical_features,
        one_hot_features=shared.one_hot_features,
        target_encoded_features=shared.target_encoded_features,
        train_missingness_report=shared.train_missingness_report,
        categorical_cardinality_report=shared.categorical_cardinality_report,
        column_contract_report=shared.column_contract_report,
    )


__all__ = [
    "Week07PreprocessorBuild",
    "build_week07_preprocessor",
    "make_categorical_cardinality_report",
    "make_feature_hash_groups",
    "make_train_missingness_report",
]
