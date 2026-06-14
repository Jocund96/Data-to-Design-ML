"""VIF diagnostics for Week 7 numeric UHPC mix features."""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression


def prepare_numeric_vif_matrix(
    X_train: pd.DataFrame,
    numeric_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare numeric training features for VIF.

    VIF is an interpretability diagnostic for OLS coefficient stability. It is
    not a model performance metric and does not use the target.
    """
    rows = []
    prepared = {}

    for feature in numeric_features:
        if feature not in X_train:
            rows.append(
                {
                    "feature": feature,
                    "status": "dropped",
                    "reason": "not_present_in_X_train",
                }
            )
            continue

        series = pd.to_numeric(X_train[feature], errors="coerce")
        if series.isna().all():
            rows.append(
                {
                    "feature": feature,
                    "status": "dropped",
                    "reason": "fully_missing",
                }
            )
            continue

        if series.nunique(dropna=True) <= 1:
            rows.append(
                {
                    "feature": feature,
                    "status": "dropped",
                    "reason": "constant_or_single_unique_value",
                }
            )
            continue

        prepared[feature] = series
        rows.append({"feature": feature, "status": "kept", "reason": ""})

    if not prepared:
        return pd.DataFrame(), pd.DataFrame(rows)

    numeric_frame = pd.DataFrame(prepared, index=X_train.index)
    imputer = SimpleImputer(strategy="median")
    imputed = pd.DataFrame(
        imputer.fit_transform(numeric_frame),
        columns=numeric_frame.columns,
        index=X_train.index,
    )

    return imputed, pd.DataFrame(rows)


def _interpret_vif(vif: float) -> str:
    """Interpret a VIF value for OLS coefficient stability."""
    if np.isinf(vif):
        return "near-perfect multicollinearity"
    if np.isnan(vif):
        return "not_enough_features"
    if vif < 5:
        return "acceptable"
    if vif < 10:
        return "high"
    return "severe"


def calculate_vif(X_numeric_imputed: pd.DataFrame) -> pd.DataFrame:
    """Calculate VIF by regressing each numeric feature on all others."""
    if X_numeric_imputed.empty:
        return pd.DataFrame(
            columns=["feature", "VIF", "R2_against_other_features", "interpretation"]
        )

    if X_numeric_imputed.shape[1] == 1:
        feature = X_numeric_imputed.columns[0]
        return pd.DataFrame(
            [
                {
                    "feature": feature,
                    "VIF": np.nan,
                    "R2_against_other_features": np.nan,
                    "interpretation": _interpret_vif(np.nan),
                }
            ]
        )

    rows = []
    for feature in X_numeric_imputed.columns:
        y_feature = X_numeric_imputed[feature]
        X_other = X_numeric_imputed.drop(columns=feature)
        model = LinearRegression()
        model.fit(X_other, y_feature)
        r2 = float(model.score(X_other, y_feature))

        if r2 >= 1 - 1e-12:
            vif = np.inf
        else:
            vif = 1 / (1 - r2)

        rows.append(
            {
                "feature": feature,
                "VIF": vif,
                "R2_against_other_features": r2,
                "interpretation": _interpret_vif(vif),
            }
        )

    return pd.DataFrame(rows).sort_values("VIF", ascending=False)


def calculate_policy_vif(
    X_train: pd.DataFrame,
    numeric_features: list[str],
    policy: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate VIF and audit tables for one feature policy."""
    X_numeric_imputed, audit = prepare_numeric_vif_matrix(X_train, numeric_features)
    vif_table = calculate_vif(X_numeric_imputed)

    if not vif_table.empty:
        vif_table.insert(0, "policy", policy)
    if not audit.empty:
        audit.insert(0, "policy", policy)

    return vif_table, audit


def summarize_vif(vif_table: pd.DataFrame) -> pd.DataFrame:
    """Summarize VIF diagnostics per policy."""
    if vif_table.empty:
        return pd.DataFrame(
            columns=[
                "policy",
                "max_vif",
                "median_vif",
                "number_features_vif_above_5",
                "number_features_vif_above_10",
                "number_inf_vif",
            ]
        )

    rows = []
    for policy, policy_df in vif_table.groupby("policy", dropna=False):
        vif_values = policy_df["VIF"]
        finite_values = vif_values.replace([np.inf, -np.inf], np.nan)
        rows.append(
            {
                "policy": policy,
                "max_vif": vif_values.max(),
                "median_vif": finite_values.median(),
                "number_features_vif_above_5": int((vif_values >= 5).sum()),
                "number_features_vif_above_10": int((vif_values >= 10).sum()),
                "number_inf_vif": int(np.isinf(vif_values).sum()),
            }
        )

    return pd.DataFrame(rows)
