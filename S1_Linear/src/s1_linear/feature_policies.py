"""
Week 6 feature policies for the UHPC semantic-missingness dataset.

The policy layer decides which non-leaking raw columns are allowed before
semantic recoding is applied.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

TARGET_COLUMN = "Mechanical Properties | Compressive Strength (Mpa) | 28-Day"

WEEK6_FEATURE_PREFIXES = (
    "Mix Constitutents |",
    "Curing Regime |",
)

NUMERIC_FEATURE_COLUMNS = (
    "Mix Constitutents | Binder | Cement Amount (kg/m³)",
    "Mix Constitutents | Binder | Cement Grade (Mpa)",
    "Mix Constitutents | Supplementary Cementitious Materials (SCMs) | Silica Fume (kg/m³)",
    "Mix Constitutents | Supplementary Cementitious Materials (SCMs) | Flayash Amount (kg/m³)",
    "Mix Constitutents | Supplementary Cementitious Materials (SCMs) | Limestone Powder (kg/m3)",
    "Mix Constitutents | Supplementary Cementitious Materials (SCMs) | Quartzpowder (kg/m3)",
    "Mix Constitutents | Supplementary Cementitious Materials (SCMs) | Glass powder (kg/m3)",
    "Mix Constitutents | Supplementary Cementitious Materials (SCMs) | Rice husk ash (kg/m3)",
    "Mix Constitutents | Supplementary Cementitious Materials (SCMs) | Metakaolin (kg/m³)",
    "Mix Constitutents | Supplementary Cementitious Materials (SCMs) | GGBFS (kg/m³)",
    "Mix Constitutents | Supplementary Cementitious Materials (SCMs) | Slag Amount (kg/m³)",
    "Mix Constitutents | Nano Particles | Nano-CaCO3 (kg/m3)",
    "Mix Constitutents | Nano Particles | Nano-Al2O3 (kg/m3)",
    "Mix Constitutents | Nano Particles | Nano-TiO2 (kg/m3)",
    "Mix Constitutents | Nano Particles | Nano Silica (kg/m3)",
    "Mix Constitutents | Sustainable Filler | Filler (kg/m³)",
    "Mix Constitutents | Sand | Amount (kg/m³)",
    "Mix Constitutents | Sand | Max Size (mm)",
    "Mix Constitutents | Fiber | Amount / Quantity of Fiber",
    "Mix Constitutents | Fiber | Length (mm)",
    "Mix Constitutents | Fiber | Diameter (mm)",
    "Mix Constitutents | Fiber | Tensile Strength (MPa)",
    "Mix Constitutents | Fiber | Nominal Young’s modulus, Gpa",
    "Mix Constitutents | Synergetic Fiber | Amount / Quantity of Fiber",
    "Mix Constitutents | Synergetic Fiber | Length (mm)",
    "Mix Constitutents | Synergetic Fiber | Diameter (mm)",
    "Mix Constitutents | Synergetic Fiber | Tensile Strength (MPa)",
    "Mix Constitutents | Synergetic Fiber | Nominal Young’s modulus, Gpa",
    "Mix Constitutents | Water | Amount (kg/m³)",
    "Mix Constitutents | Superplasticizer | Amount (kg/m³)",
    "Curing Regime | Temperature (o C)",
    "Curing Regime | Humidity (%)",
    "Curing Regime | Pressure (MPa)",
)

CATEGORICAL_MISSING_VALUE = "missing_reported_gap"

LOW_CARDINALITY_MAX = 10
MEDIUM_CARDINALITY_MAX = 20


@dataclass(frozen=True)
class Week6PreprocessorBuild:
    """A fitted-ready preprocessor plus train-derived preprocessing reports."""

    preprocessor: object
    numeric_missing_audit: pd.DataFrame
    cardinality_report: pd.DataFrame
    preprocessing_summary: pd.DataFrame
    dropped_numeric_features: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    low_cardinality_features: list[str]
    medium_cardinality_features: list[str]
    high_cardinality_features: list[str]


class DomainCategoryGrouper:
    """
    Collapse common UHPC category variants before rare-category encoding.

    Values that do not match a domain pattern are normalized but preserved so
    the downstream encoder can still keep frequent values and group rare ones.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            frame = X.copy()
        else:
            frame = pd.DataFrame(X)

        return frame.apply(lambda column: column.map(self._group_value)).to_numpy(
            dtype=object
        )

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([], dtype=object)
        return np.asarray(input_features, dtype=object)

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self

    @staticmethod
    def _normalize(value) -> str:
        text = str(value).strip().casefold()
        for char in ["_", "-", "/", "\\", "|", ",", ";", "(", ")", "[", "]"]:
            text = text.replace(char, " ")
        return " ".join(text.split())

    @classmethod
    def _group_value(cls, value):
        if pd.isna(value):
            return CATEGORICAL_MISSING_VALUE

        text = cls._normalize(value)
        compact = text.replace(" ", "")
        tokens = set(text.split())

        if not text:
            return CATEGORICAL_MISSING_VALUE
        if text in {"missing reported gap", CATEGORICAL_MISSING_VALUE}:
            return CATEGORICAL_MISSING_VALUE
        if text in {"not applicable", "not_applicable", "none", "no", "nil"}:
            return "not_applicable"
        if text in {"unknown type", "unknown_type", "unknown", "not reported"}:
            return "unknown_type"

        if "polycarboxylate" in text or "pce" in tokens:
            return "superplasticizer_polycarboxylate"
        if "naphthalene" in text or "snf" in tokens:
            return "superplasticizer_naphthalene"
        if "melamine" in text or "smf" in tokens:
            return "superplasticizer_melamine"
        if "lignosulfonate" in text or "lignosulphonate" in text:
            return "superplasticizer_lignosulfonate"

        if "steel" in text:
            return "fiber_steel"
        if "polypropylene" in text or tokens & {"pp"}:
            return "fiber_polypropylene"
        if "polyethylene" in text or tokens & {"pe", "hdpe", "uhmwpe"}:
            return "fiber_polyethylene"
        if "polyvinyl alcohol" in text or tokens & {"pva"}:
            return "fiber_pva"
        if "carbon" in text:
            return "fiber_carbon"
        if "glass" in text:
            return "fiber_glass"
        if "basalt" in text:
            return "fiber_basalt"

        if "cem i" in text or "portland" in text or tokens & {"opc"}:
            return "cement_portland"
        if "cem ii" in text or "blended cement" in text:
            return "cement_blended"
        if "white cement" in text:
            return "cement_white"
        if "calcium aluminate" in text:
            return "cement_calcium_aluminate"
        if "sulfoaluminate" in text or "sulphoaluminate" in text:
            return "cement_sulfoaluminate"

        if text in {"f", "class f", "fly ash class f"} or "class f" in text:
            return "fly_ash_class_f"
        if text in {"c", "class c", "fly ash class c"} or "class c" in text:
            return "fly_ash_class_c"

        if "ggbfs" in compact or "ground granulated" in text:
            return "slag_ggbfs"
        if "blast furnace" in text or "slag" in text:
            return "slag_blast_furnace"

        if "quartz" in text:
            return "sand_or_filler_quartz"
        if "silica" in text:
            return "sand_or_filler_silica"
        if "river" in text:
            return "sand_river"
        if "limestone" in text or "calcium carbonate" in text or "caco3" in compact:
            return "filler_limestone"
        if "rice husk" in text:
            return "scm_rice_husk_ash"
        if "metakaolin" in text:
            return "scm_metakaolin"

        return text.replace(" ", "_")


def select_week6_candidate_features(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
) -> list[str]:
    """Selecting mix constituent and curing columns as non-leaking candidates."""
    return [
        column
        for column in df.columns
        if column != target_col and column.startswith(WEEK6_FEATURE_PREFIXES)
    ]


def classify_excluded_column(column: str, target_col: str = TARGET_COLUMN) -> str:
    """Explain why a column is not used as a Week 6 predictor."""
    if column == target_col:
        return "target_column"
    if column == "Mix-ID":
        return "identifier"
    if column.startswith("Workability |"):
        return "workability_excluded_by_week6_scope"
    if column.startswith("Mechanical Properties |"):
        return "mechanical_outcome_or_testing_metadata_leakage"
    if column.startswith("Durability Properities |"):
        return "durability_outcome_leakage"
    if column.startswith("Research Paper Details |"):
        return "paper_metadata"
    if column.startswith(WEEK6_FEATURE_PREFIXES):
        return "candidate_feature"
    return "outside_week6_scope"


def make_excluded_columns_table(
    df: pd.DataFrame,
    selected_features: list[str],
    target_col: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Create an audit table for columns excluded before modeling."""
    selected = set(selected_features)
    rows = []

    for column in df.columns:
        if column in selected or column == target_col:
            continue

        rows.append(
            {
                "column": column,
                "reason": classify_excluded_column(column, target_col=target_col),
            }
        )

    return pd.DataFrame(rows)


def get_numeric_feature_columns(columns: list[str] | pd.Index) -> list[str]:
    """Return known numeric UHPC feature columns that are present."""
    available = set(columns)
    return [column for column in NUMERIC_FEATURE_COLUMNS if column in available]


def get_categorical_feature_columns(columns: list[str] | pd.Index) -> list[str]:
    """Return selected feature columns that are not in the numeric list."""
    numeric = set(get_numeric_feature_columns(columns))
    return [column for column in columns if column not in numeric]


def detect_week6_feature_types(
    X_train: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Detect numeric and categorical Week 6 columns from the train split."""
    known_numeric = set(get_numeric_feature_columns(X_train.columns))
    dtype_numeric = set(X_train.select_dtypes(include=["number"]).columns)
    numeric = [
        column
        for column in X_train.columns
        if column in known_numeric or column in dtype_numeric
    ]
    categorical = [column for column in X_train.columns if column not in set(numeric)]
    return numeric, categorical


def coerce_numeric_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert known numeric UHPC feature columns to numeric dtype.

    The report captures where text-like values became NaN during coercion.
    """
    df = df.copy()
    rows = []

    for column in get_numeric_feature_columns(df.columns):
        missing_before = int(df[column].isna().sum())
        df[column] = pd.to_numeric(df[column], errors="coerce")
        missing_after = int(df[column].isna().sum())

        rows.append(
            {
                "column": column,
                "missing_before_numeric_coercion": missing_before,
                "missing_after_numeric_coercion": missing_after,
                "new_missing_from_numeric_coercion": missing_after - missing_before,
            }
        )

    return df, pd.DataFrame(rows)


def filter_features_by_missingness(
    df: pd.DataFrame,
    features: list[str],
    missing_threshold: float | None,
) -> tuple[list[str], pd.DataFrame]:
    """
    Apply a raw-column missingness threshold.

    A threshold of 0.50 keeps columns with at most 50 percent missing values.
    None keeps all candidate features.
    """
    missing_fraction = df[features].isna().mean()
    rows = []
    kept_features = []

    for feature in features:
        fraction = float(missing_fraction[feature])
        keep = missing_threshold is None or fraction <= missing_threshold

        if keep:
            kept_features.append(feature)

        rows.append(
            {
                "feature": feature,
                "missing_fraction_before_semantic_recode": fraction,
                "missing_percentage_before_semantic_recode": fraction * 100,
                "missing_threshold": missing_threshold,
                "kept": keep,
                "drop_reason": (
                    "" if keep else f"missing_fraction_above_{missing_threshold}"
                ),
            }
        )

    return kept_features, pd.DataFrame(rows)


def make_numeric_fully_missing_audit(
    X_train: pd.DataFrame,
    numeric_features: list[str],
) -> tuple[list[str], pd.DataFrame]:
    """Identify numeric columns that are unusable because train is all missing."""
    rows = []
    kept_features = []
    train_rows = len(X_train)

    for column in numeric_features:
        missing_count = int(X_train[column].isna().sum())
        fully_missing = missing_count == train_rows

        if not fully_missing:
            kept_features.append(column)

        rows.append(
            {
                "column": column,
                "train_rows": train_rows,
                "missing_count_train": missing_count,
                "missing_fraction_train": (
                    missing_count / train_rows if train_rows else np.nan
                ),
                "fully_missing_in_train": fully_missing,
                "dropped_from_modeling": fully_missing,
                "drop_reason": (
                    "numeric_fully_missing_in_X_train" if fully_missing else ""
                ),
            }
        )

    return kept_features, pd.DataFrame(
        rows,
        columns=[
            "column",
            "train_rows",
            "missing_count_train",
            "missing_fraction_train",
            "fully_missing_in_train",
            "dropped_from_modeling",
            "drop_reason",
        ],
    )


def make_categorical_cardinality_report(
    X_train: pd.DataFrame,
    categorical_features: list[str],
    low_cardinality_max: int = LOW_CARDINALITY_MAX,
    medium_cardinality_max: int = MEDIUM_CARDINALITY_MAX,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """Count train-only category cardinality and assign encoder buckets."""
    rows = []
    low_cardinality_features = []
    medium_cardinality_features = []
    high_cardinality_features = []

    for column in categorical_features:
        filled = X_train[column].fillna(CATEGORICAL_MISSING_VALUE).astype(str)
        unique_categories = int(filled.nunique(dropna=False))
        missing_count = int(X_train[column].isna().sum())

        if unique_categories <= low_cardinality_max:
            bucket = "low"
            encoder_policy = "onehot_ignore_unknown"
            low_cardinality_features.append(column)
        elif unique_categories <= medium_cardinality_max:
            bucket = "medium"
            encoder_policy = "rare_category_onehot"
            medium_cardinality_features.append(column)
        else:
            bucket = "high"
            encoder_policy = "domain_grouping_then_rare_category_onehot"
            high_cardinality_features.append(column)

        rows.append(
            {
                "column": column,
                "missing_count_train": missing_count,
                "missing_fraction_train": (
                    missing_count / len(X_train) if len(X_train) else np.nan
                ),
                "unique_categories_train": unique_categories,
                "cardinality_bucket": bucket,
                "encoder_policy": encoder_policy,
                "low_cardinality_max": low_cardinality_max,
                "medium_cardinality_max": medium_cardinality_max,
            }
        )

    return (
        pd.DataFrame(
            rows,
            columns=[
                "column",
                "missing_count_train",
                "missing_fraction_train",
                "unique_categories_train",
                "cardinality_bucket",
                "encoder_policy",
                "low_cardinality_max",
                "medium_cardinality_max",
            ],
        ),
        low_cardinality_features,
        medium_cardinality_features,
        high_cardinality_features,
    )


def _make_one_hot_encoder(
    handle_unknown: str,
    min_frequency: int | None = None,
    max_categories: int | None = None,
):
    from sklearn.preprocessing import OneHotEncoder

    kwargs = {"handle_unknown": handle_unknown}
    if min_frequency is not None:
        kwargs["min_frequency"] = min_frequency
    if max_categories is not None:
        kwargs["max_categories"] = max_categories

    for sparse_kwargs in ({"sparse_output": False}, {"sparse": False}):
        try:
            return OneHotEncoder(**kwargs, **sparse_kwargs)
        except TypeError:
            continue

    if handle_unknown == "infrequent_if_exist":
        return _make_one_hot_encoder(handle_unknown="ignore")

    return OneHotEncoder(handle_unknown=handle_unknown)


def _make_numeric_imputer(kind: str, add_indicator: bool, knn_neighbors: int):
    from sklearn.impute import KNNImputer, SimpleImputer

    if kind == "simple":
        return SimpleImputer(strategy="median", add_indicator=add_indicator)
    if kind == "knn":
        return KNNImputer(n_neighbors=knn_neighbors, add_indicator=add_indicator)

    raise ValueError("numeric_imputer must be either 'simple' or 'knn'.")


def _make_categorical_missing_imputer():
    from sklearn.impute import SimpleImputer

    return SimpleImputer(
        strategy="constant",
        fill_value=CATEGORICAL_MISSING_VALUE,
    )


def _make_target_encoder(random_state: int | None):
    try:
        from sklearn.preprocessing import TargetEncoder
    except ImportError as exc:
        raise ImportError(
            "TargetEncoder is unavailable in this scikit-learn installation. "
            "Disable the target_encoder_high_cardinality preprocessing policy "
            "or upgrade scikit-learn."
        ) from exc

    try:
        return TargetEncoder(target_type="continuous", random_state=random_state)
    except TypeError:
        return TargetEncoder()


def build_week6_preprocessor(
    X_train: pd.DataFrame,
    y_train: pd.Series | None = None,
    numeric_imputer: str = "simple",
    numeric_add_indicator: bool = True,
    knn_neighbors: int = 5,
    low_cardinality_max: int = LOW_CARDINALITY_MAX,
    medium_cardinality_max: int = MEDIUM_CARDINALITY_MAX,
    high_cardinality_policy: str = "grouped_ohe",
    random_state: int | None = None,
) -> Week6PreprocessorBuild:
    """
    Build a leakage-safe preprocessing object from X_train and y_train only.

    Validation and test sets should only call transform with the fitted object.
    A TargetEncoder policy requires y_train and should be fitted with y_train.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    numeric_features, categorical_features = detect_week6_feature_types(X_train)
    numeric_features, numeric_missing_audit = make_numeric_fully_missing_audit(
        X_train=X_train,
        numeric_features=numeric_features,
    )
    dropped_mask = numeric_missing_audit["dropped_from_modeling"].astype(bool)
    dropped_numeric_features = numeric_missing_audit.loc[
        dropped_mask, "column"
    ].tolist()

    (
        cardinality_report,
        low_cardinality_features,
        medium_cardinality_features,
        high_cardinality_features,
    ) = make_categorical_cardinality_report(
        X_train=X_train,
        categorical_features=categorical_features,
        low_cardinality_max=low_cardinality_max,
        medium_cardinality_max=medium_cardinality_max,
    )

    if high_cardinality_policy not in {"grouped_ohe", "target_encoder"}:
        raise ValueError(
            "high_cardinality_policy must be 'grouped_ohe' or 'target_encoder'."
        )
    if high_cardinality_policy == "target_encoder" and high_cardinality_features:
        if y_train is None:
            raise ValueError(
                "y_train is required when high_cardinality_policy='target_encoder'."
            )
        if len(y_train) != len(X_train):
            raise ValueError("X_train and y_train must have the same number of rows.")

    transformers = []

    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "imputer",
                            _make_numeric_imputer(
                                kind=numeric_imputer,
                                add_indicator=numeric_add_indicator,
                                knn_neighbors=knn_neighbors,
                            ),
                        ),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )

    if low_cardinality_features:
        transformers.append(
            (
                "categorical_low",
                Pipeline(
                    [
                        ("imputer", _make_categorical_missing_imputer()),
                        ("onehot", _make_one_hot_encoder(handle_unknown="ignore")),
                    ]
                ),
                low_cardinality_features,
            )
        )

    if medium_cardinality_features:
        transformers.append(
            (
                "categorical_medium",
                Pipeline(
                    [
                        ("imputer", _make_categorical_missing_imputer()),
                        (
                            "onehot",
                            _make_one_hot_encoder(
                                handle_unknown="infrequent_if_exist",
                                min_frequency=10,
                                max_categories=20,
                            ),
                        ),
                    ]
                ),
                medium_cardinality_features,
            )
        )

    if high_cardinality_features and high_cardinality_policy == "grouped_ohe":
        transformers.append(
            (
                "categorical_high_grouped_ohe",
                Pipeline(
                    [
                        ("imputer", _make_categorical_missing_imputer()),
                        ("domain_grouper", DomainCategoryGrouper()),
                        (
                            "onehot",
                            _make_one_hot_encoder(
                                handle_unknown="infrequent_if_exist",
                                min_frequency=10,
                                max_categories=20,
                            ),
                        ),
                    ]
                ),
                high_cardinality_features,
            )
        )

    if high_cardinality_features and high_cardinality_policy == "target_encoder":
        transformers.append(
            (
                "categorical_high_target_encoder",
                Pipeline(
                    [
                        ("imputer", _make_categorical_missing_imputer()),
                        ("domain_grouper", DomainCategoryGrouper()),
                        ("target_encoder", _make_target_encoder(random_state)),
                    ]
                ),
                high_cardinality_features,
            )
        )

    summary = pd.DataFrame(
        [
            {
                "preprocessing_policy": high_cardinality_policy,
                "input_features": X_train.shape[1],
                "numeric_features_detected": len(numeric_missing_audit),
                "numeric_features_dropped_fully_missing_train": len(
                    dropped_numeric_features
                ),
                "numeric_features_used": len(numeric_features),
                "categorical_features_detected": len(categorical_features),
                "low_cardinality_features": len(low_cardinality_features),
                "medium_cardinality_features": len(medium_cardinality_features),
                "high_cardinality_features": len(high_cardinality_features),
                "numeric_imputer": numeric_imputer,
                "numeric_missing_indicators": numeric_add_indicator,
                "knn_neighbors": knn_neighbors if numeric_imputer == "knn" else np.nan,
                "numeric_scaler": "StandardScaler",
                "categorical_missing_fill_value": CATEGORICAL_MISSING_VALUE,
                "low_cardinality_encoder": "OneHotEncoder(handle_unknown='ignore')",
                "medium_cardinality_encoder": (
                    "OneHotEncoder(handle_unknown='infrequent_if_exist', "
                    "min_frequency=10, max_categories=20)"
                ),
                "high_cardinality_encoder": (
                    "TargetEncoder"
                    if high_cardinality_policy == "target_encoder"
                    else (
                        "DomainCategoryGrouper + "
                        "OneHotEncoder(handle_unknown='infrequent_if_exist', "
                        "min_frequency=10, max_categories=20)"
                    )
                ),
                "target_encoder_fitted_with_y_train": (
                    high_cardinality_policy == "target_encoder"
                    and bool(high_cardinality_features)
                ),
            }
        ]
    )

    return Week6PreprocessorBuild(
        preprocessor=ColumnTransformer(transformers=transformers, remainder="drop"),
        numeric_missing_audit=numeric_missing_audit,
        cardinality_report=cardinality_report,
        preprocessing_summary=summary,
        dropped_numeric_features=dropped_numeric_features,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        low_cardinality_features=low_cardinality_features,
        medium_cardinality_features=medium_cardinality_features,
        high_cardinality_features=high_cardinality_features,
    )
