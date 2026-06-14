"""
Week 6 semantic missingness utilities for the UHPC dataset.

The key idea is that a blank value is not always the same kind of missingness.
For amount/type material pairs, a blank type can mean either "not used" or
"used, but the type was not reported" depending on the amount column.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


UNKNOWN_TYPE = "unknown_type"
NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class AmountTypePair:
    """A material amount column and its matching material type column."""

    label: str
    amount_col: str
    type_col: str


DEFAULT_AMOUNT_TYPE_PAIRS = [
    AmountTypePair(
        label="cement",
        amount_col="Mix Constitutents | Binder | Cement Amount (kg/m³)",
        type_col="Mix Constitutents | Binder | Cement type",
    ),
    AmountTypePair(
        label="fly_ash",
        amount_col=(
            "Mix Constitutents | Supplementary Cementitious Materials (SCMs) | "
            "Flayash Amount (kg/m³)"
        ),
        type_col=(
            "Mix Constitutents | Supplementary Cementitious Materials (SCMs) | "
            "Fly Ash Type"
        ),
    ),
    AmountTypePair(
        label="slag",
        amount_col=(
            "Mix Constitutents | Supplementary Cementitious Materials (SCMs) | "
            "Slag Amount (kg/m³)"
        ),
        type_col=(
            "Mix Constitutents | Supplementary Cementitious Materials (SCMs) | "
            "Type of Slag"
        ),
    ),
    AmountTypePair(
        label="sustainable_filler",
        amount_col="Mix Constitutents | Sustainable Filler | Filler (kg/m³)",
        type_col="Mix Constitutents | Sustainable Filler | Type of Filler",
    ),
    AmountTypePair(
        label="sand",
        amount_col="Mix Constitutents | Sand | Amount (kg/m³)",
        type_col="Mix Constitutents | Sand | Sand Type",
    ),
    AmountTypePair(
        label="fiber",
        amount_col="Mix Constitutents | Fiber | Amount / Quantity of Fiber",
        type_col="Mix Constitutents | Fiber | Type of Fiber",
    ),
    AmountTypePair(
        label="synergetic_fiber",
        amount_col=(
            "Mix Constitutents | Synergetic Fiber | Amount / Quantity of Fiber"
        ),
        type_col="Mix Constitutents | Synergetic Fiber | Type of Fiber",
    ),
    AmountTypePair(
        label="superplasticizer",
        amount_col="Mix Constitutents | Superplasticizer | Amount (kg/m³)",
        type_col=(
            "Mix Constitutents | Superplasticizer | Type of Superplasticizer"
        ),
    ),
]


MISSING_TEXT_VALUES = [
    "",
    " ",
    "NA",
    "N/A",
    "na",
    "n/a",
    "None",
    "none",
    "null",
    "Null",
    "nan",
    "NaN",
]


def normalize_text_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip text columns and convert common blank markers to real NaN values.

    Pandas already reads empty CSV cells as NaN in many cases, but this function
    also handles whitespace-only cells and text markers such as "N/A".
    """
    df = df.copy()
    text_columns = df.select_dtypes(include=["object"]).columns

    for column in text_columns:
        df[column] = df[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )

    return df.replace(MISSING_TEXT_VALUES, np.nan)


def recode_semantic_missingness(
    df: pd.DataFrame,
    pairs: list[AmountTypePair] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Recode semantic missingness for material amount/type pairs.

    Rules:
    - amount > 0 and type missing: type = unknown_type
    - amount missing and type present: amount stays missing
    - amount missing or amount == 0 and type missing:
      amount = 0 and type = not_applicable

    Returns:
        recoded_df: DataFrame after semantic recoding
        summary_df: before/after counts for every configured pair
    """
    df = normalize_text_missing_values(df)
    pairs = pairs or DEFAULT_AMOUNT_TYPE_PAIRS
    summary_rows = []

    for pair in pairs:
        if pair.amount_col not in df.columns or pair.type_col not in df.columns:
            summary_rows.append(
                {
                    "pair": pair.label,
                    "amount_col": pair.amount_col,
                    "type_col": pair.type_col,
                    "applied": False,
                    "reason": "amount_or_type_column_not_in_policy",
                }
            )
            continue

        amount = pd.to_numeric(df[pair.amount_col], errors="coerce")
        df[pair.amount_col] = amount

        type_missing = df[pair.type_col].isna()
        amount_missing = amount.isna()
        amount_positive = amount.gt(0)
        amount_zero_or_missing = amount_missing | amount.eq(0)

        used_type_unknown = amount_positive & type_missing
        not_used_type_missing = amount_zero_or_missing & type_missing
        type_present_amount_unknown = amount_missing & ~type_missing

        before_amount_missing = int(amount_missing.sum())
        before_type_missing = int(type_missing.sum())
        before_both_missing = int((amount_missing & type_missing).sum())

        df.loc[used_type_unknown, pair.type_col] = UNKNOWN_TYPE
        df.loc[not_used_type_missing, pair.amount_col] = 0
        df.loc[not_used_type_missing, pair.type_col] = NOT_APPLICABLE

        after_amount_missing = int(df[pair.amount_col].isna().sum())
        after_type_missing = int(df[pair.type_col].isna().sum())

        summary_rows.append(
            {
                "pair": pair.label,
                "amount_col": pair.amount_col,
                "type_col": pair.type_col,
                "applied": True,
                "reason": "semantic_rules_applied",
                "amount_missing_before": before_amount_missing,
                "type_missing_before": before_type_missing,
                "both_missing_before": before_both_missing,
                "amount_positive_type_missing_to_unknown_type": int(
                    used_type_unknown.sum()
                ),
                "amount_zero_or_missing_type_missing_to_not_applicable": int(
                    not_used_type_missing.sum()
                ),
                "type_present_amount_missing_left_missing": int(
                    type_present_amount_unknown.sum()
                ),
                "amount_missing_after": after_amount_missing,
                "type_missing_after": after_type_missing,
                "amount_zero_after": int(df[pair.amount_col].eq(0).sum()),
                "unknown_type_after": int((df[pair.type_col] == UNKNOWN_TYPE).sum()),
                "not_applicable_after": int(
                    (df[pair.type_col] == NOT_APPLICABLE).sum()
                ),
            }
        )

    return df, pd.DataFrame(summary_rows)


def make_missing_before_after_report(
    before: pd.DataFrame,
    after: pd.DataFrame,
) -> pd.DataFrame:
    """Create a column-level missingness report before and after recoding."""
    rows = []
    columns = [column for column in before.columns if column in after.columns]

    for column in columns:
        missing_before = int(before[column].isna().sum())
        missing_after = int(after[column].isna().sum())
        rows.append(
            {
                "column": column,
                "missing_before": missing_before,
                "missing_after": missing_after,
                "missing_reduced_by": missing_before - missing_after,
                "missing_percentage_before": before[column].isna().mean() * 100,
                "missing_percentage_after": after[column].isna().mean() * 100,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["missing_percentage_before", "column"],
        ascending=[False, True],
    )
