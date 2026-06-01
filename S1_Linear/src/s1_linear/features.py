"""
Feature engineering utilities for Week 4 representation experiments.

The goal for this Week is to test whether different feature representations
improve the Linear Family models.

I have created:
1. Yeh/domain-inspired engineered features
2. Log-transformed features for skewed variables
3. More aggressive interaction features
"""

import numpy as np
import pandas as pd

EPSILON = 1e-8


def add_yeh_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adding concrete-domain engineered features.

    These features are calculated from the original UCI/Yeh columns.

    Important:
    - They are not completely new raw information.
    - They are a better representation of existing information.
    - Linear models often benefit from ratios because they cannot easily
      learn division relationships by themselves.

    Returns:
        DataFrame with additional engineered features.
    """
    df = df.copy()

    df["TotalBinder"] = df["Cement"] + df["Slag"] + df["FlyAsh"]

    df["WaterToCement"] = df["Water"] / (df["Cement"] + EPSILON)
    df["WaterToBinder"] = df["Water"] / (df["TotalBinder"] + EPSILON)

    df["SPToBinder"] = df["Superplasticizer"] / (df["TotalBinder"] + EPSILON)
    df["FlyAshToBinder"] = df["FlyAsh"] / (df["TotalBinder"] + EPSILON)
    df["SlagToBinder"] = df["Slag"] / (df["TotalBinder"] + EPSILON)

    return df


def add_log_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adding log-transformed features.

    log1p(x) means log(1 + x), so it also works when x = 0.

    These transformations are useful for skewed features.
    In concrete data, Age is a good example because strength increases
    quickly at early age and more slowly later.
    """
    df = df.copy()

    df["LogAge"] = np.log1p(df["Age"])
    df["LogSuperplasticizer"] = np.log1p(df["Superplasticizer"])
    df["LogSlag"] = np.log1p(df["Slag"])
    df["LogFlyAsh"] = np.log1p(df["FlyAsh"])

    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adding more aggressive interaction features.

    These features test whether combinations between variables help
    the linear models.

    Example:
    WaterToBinder * LogAge can represent that the effect of age may depend
    on the water/binder ratio.
    """
    df = df.copy()

    if "TotalBinder" not in df.columns:
        df = add_yeh_engineered_features(df)

    if "LogAge" not in df.columns:
        df = add_log_features(df)

    df["BinderAge"] = df["TotalBinder"] * df["LogAge"]
    df["WaterBinderAge"] = df["WaterToBinder"] * df["LogAge"]
    df["CementAge"] = df["Cement"] * df["LogAge"]
    df["SPAge"] = df["Superplasticizer"] * df["LogAge"]

    df["CementWaterInteraction"] = df["Cement"] * df["Water"]
    df["BinderWaterInteraction"] = df["TotalBinder"] * df["Water"]

    return df


def build_feature_representation(df: pd.DataFrame, representation: str) -> pd.DataFrame:
    """
    Building the selected Week 4 feature representation.

    Available representations:
    - original
    - yeh_engineered
    - log_transformed
    - aggressive_interactions
    """

    df = df.copy()

    if representation == "original":
        return df

    if representation == "yeh_engineered":
        return add_yeh_engineered_features(df)

    if representation == "log_transformed":
        df = add_yeh_engineered_features(df)
        df = add_log_features(df)
        return df

    if representation == "aggressive_interactions":
        df = add_yeh_engineered_features(df)
        df = add_log_features(df)
        df = add_interaction_features(df)
        return df

    raise ValueError(f"Unknown representation: {representation}")
