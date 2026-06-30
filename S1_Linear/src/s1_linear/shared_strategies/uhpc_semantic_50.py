"""Shared UHPC semantic-recoded 50% preprocessing contract.

From Week 7 onward, the Linear Family work uses the shared semantic-recoded
UHPC 50% dataset and this fixed preprocessing contract so the transformed
feature count is comparable across model families.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder


SHARED_NUMERIC_FEATURES = [
    "cement",
    "silica_fume",
    "fly_ash",
    "limestone_powder",
    "quartz_powder",
    "glass_powder",
    "rice_husk_ash",
    "metakaolin",
    "ggbfs",
    "slag",
    "nano_caco3",
    "nano_al2o3",
    "nano_tio2",
    "nano_sio2",
    "filler",
    "sand",
    "sand_max_size",
    "fiber1_amount",
    "fiber1_length",
    "fiber1_diameter",
    "fiber2_amount",
    "water",
    "sp_amount",
    "curing_temp",
]

SHARED_ONE_HOT_FEATURES = [
    "fly_ash_type",
    "slag_type",
    "fiber2_type",
    "sp_type",
    "curing_method",
]

SHARED_TARGET_ENCODED_FEATURES = [
    "cement_type",
    "filler_type",
    "sand_type",
    "fiber1_type",
]

TARGET_COLUMN = "cs_28d"
PUBLICATION_COLUMN = "paper_reference"
EXPECTED_TRANSFORMED_COLUMNS = 60


@dataclass(frozen=True)
class SharedPreprocessorBuild:
    """Preprocessor and reports for the shared UHPC strategy."""

    preprocessor: Pipeline
    numeric_features: list[str]
    one_hot_features: list[str]
    target_encoded_features: list[str]
    categorical_features: list[str]
    train_missingness_report: pd.DataFrame
    categorical_cardinality_report: pd.DataFrame
    column_contract_report: pd.DataFrame


def _present(columns: list[str], X: pd.DataFrame) -> list[str]:
    """Return configured columns that are available in X, preserving order."""
    return [column for column in columns if column in X.columns]


def _resolve_categorical_groups(
    X_train: pd.DataFrame,
    one_hot_features: list[str] | None,
    target_encoded_features: list[str] | None,
    categorical_features: list[str] | None,
) -> tuple[list[str], list[str]]:
    """Resolve one-hot and target-encoded groups from fixed shared defaults."""
    if one_hot_features is not None or target_encoded_features is not None:
        one_hot = _present(one_hot_features or [], X_train)
        target = _present(target_encoded_features or [], X_train)
        return one_hot, target

    available_categorical = (
        _present(categorical_features, X_train)
        if categorical_features is not None
        else X_train.select_dtypes(exclude=["number"]).columns.tolist()
    )
    available_set = set(available_categorical)

    one_hot = [
        column for column in SHARED_ONE_HOT_FEATURES if column in available_set
    ]
    target = [
        column
        for column in SHARED_TARGET_ENCODED_FEATURES
        if column in available_set
    ]

    assigned = set(one_hot) | set(target)
    for column in available_categorical:
        if column in assigned:
            continue
        if X_train[column].nunique(dropna=True) <= 10:
            one_hot.append(column)
        else:
            target.append(column)

    return one_hot, target


def normalize_shared_uhpc_semantic_50(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    keep_publication: bool = False,
) -> pd.DataFrame:
    """
    Normalize the shared semantic-recoded 50% UHPC table for modeling.

    The source semantic file still contains ``cement_grade`` and missing
    first-fiber geometry values. The shared modeling setup drops
    ``cement_grade`` and sets missing ``fiber1_length``/``fiber1_diameter`` to
    zero, matching the publication-aware shared file used for Week 8+.
    """
    normalized = df.copy()
    artifact_columns = [
        column for column in ["Unnamed: 0", "cement_type_clean"] if column in normalized
    ]
    if artifact_columns:
        normalized = normalized.drop(columns=artifact_columns)
    if "cement_grade" in normalized:
        normalized = normalized.drop(columns=["cement_grade"])
    if not keep_publication and PUBLICATION_COLUMN in normalized:
        normalized = normalized.drop(columns=[PUBLICATION_COLUMN])

    for column in ["fiber1_length", "fiber1_diameter"]:
        if column in normalized:
            normalized[column] = pd.to_numeric(
                normalized[column],
                errors="coerce",
            ).fillna(0)

    if target_col not in normalized:
        raise ValueError(f"Target column not found in shared UHPC data: {target_col}")
    normalized[target_col] = pd.to_numeric(normalized[target_col], errors="raise")
    return normalized


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
    one_hot_features: list[str],
    target_encoded_features: list[str],
) -> pd.DataFrame:
    """Profile categorical cardinality from the training split only."""
    rows = []
    for encoding, columns in [
        ("OneHotEncoder(handle_unknown='ignore')", one_hot_features),
        ("TargetEncoder(cv=5)", target_encoded_features),
    ]:
        for column in columns:
            filled = X_train[column].astype("object")
            rows.append(
                {
                    "column": column,
                    "encoding_group": (
                        "one_hot" if encoding.startswith("OneHot") else "target"
                    ),
                    "missing_count_train": int(X_train[column].isna().sum()),
                    "unique_categories_train": int(filled.nunique(dropna=True)),
                    "encoder": encoding,
                }
            )
    return pd.DataFrame(rows)


def make_shared_column_contract_report(
    X_train: pd.DataFrame,
    numeric_features: list[str],
    one_hot_features: list[str],
    target_encoded_features: list[str],
    transformed_columns: int | None = None,
) -> pd.DataFrame:
    """Document how raw shared features map to transformed model inputs."""
    rows = []
    for group_name, columns in [
        ("numeric_scaled", numeric_features),
        ("one_hot_encoded", one_hot_features),
        ("target_encoded_scaled", target_encoded_features),
    ]:
        for column in columns:
            rows.append(
                {
                    "source_column": column,
                    "shared_feature_group": group_name,
                    "dtype_in_training": str(X_train[column].dtype),
                    "missing_count_train": int(X_train[column].isna().sum()),
                    "unique_values_train": int(X_train[column].nunique(dropna=True)),
                    "final_transformed_columns_total": transformed_columns,
                }
            )
    return pd.DataFrame(rows)


def _target_encoder(random_state: int | None) -> TargetEncoder:
    """Create TargetEncoder while staying compatible across sklearn versions."""
    kwargs = {"cv": 5}
    if random_state is not None:
        kwargs["random_state"] = random_state
    try:
        return TargetEncoder(target_type="continuous", **kwargs)
    except TypeError:
        return TargetEncoder(**kwargs)


def build_shared_uhpc_preprocessor(
    X_train: pd.DataFrame,
    numeric_features: list[str] | None = None,
    one_hot_features: list[str] | None = None,
    target_encoded_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    random_state: int | None = 42,
) -> SharedPreprocessorBuild:
    """
    Build the shared UHPC preprocessor fitted later on training rows only.

    Numeric columns use median imputation and StandardScaler. Low-cardinality
    shared categoricals use one-hot encoding. Higher-cardinality shared
    categoricals use sklearn TargetEncoder with internal CV, then those encoded
    columns are scaled while one-hot outputs remain 0/1.
    """
    numeric = (
        _present(numeric_features, X_train)
        if numeric_features is not None
        else _present(SHARED_NUMERIC_FEATURES, X_train)
    )
    one_hot, target = _resolve_categorical_groups(
        X_train=X_train,
        one_hot_features=one_hot_features,
        target_encoded_features=target_encoded_features,
        categorical_features=categorical_features,
    )
    categorical = one_hot + target

    if not numeric:
        raise ValueError("No numeric predictors were found for shared preprocessing.")

    transformers = [
        (
            "num",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric,
        )
    ]
    if one_hot:
        transformers.append(
            (
                "ohe",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                one_hot,
            )
        )
    if target:
        transformers.append(
            (
                "target",
                _target_encoder(random_state=random_state),
                target,
            )
        )

    column_transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )

    steps = [("preprocessor", column_transformer)]
    if target:
        steps.append(
            (
                "scale_target_encoded",
                ColumnTransformer(
                    [
                        (
                            "scale_target",
                            StandardScaler(),
                            list(range(-len(target), 0)),
                        )
                    ],
                    remainder="passthrough",
                    verbose_feature_names_out=False,
                ),
            )
        )

    preprocessor = Pipeline(steps)
    contract_report = make_shared_column_contract_report(
        X_train=X_train,
        numeric_features=numeric,
        one_hot_features=one_hot,
        target_encoded_features=target,
    )
    return SharedPreprocessorBuild(
        preprocessor=preprocessor,
        numeric_features=numeric,
        one_hot_features=one_hot,
        target_encoded_features=target,
        categorical_features=categorical,
        train_missingness_report=make_train_missingness_report(X_train),
        categorical_cardinality_report=make_categorical_cardinality_report(
            X_train=X_train,
            one_hot_features=one_hot,
            target_encoded_features=target,
        ),
        column_contract_report=contract_report,
    )
