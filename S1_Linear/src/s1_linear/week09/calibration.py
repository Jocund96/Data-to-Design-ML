"""Prediction-interval construction and calibration metrics for Week 9."""

import math

import numpy as np
import pandas as pd

from s1_linear.week07_metrics import regression_metrics


def finite_sample_conformal_quantile(scores, alpha: float) -> float:
    """Return the finite-sample split-conformal higher order statistic."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1.")
    values = np.asarray(scores, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("At least one finite calibration score is required.")
    rank = min(len(values), math.ceil((len(values) + 1) * (1 - alpha)))
    return float(np.partition(values, rank - 1)[rank - 1])


def conformal_quantile_audit(scores, alpha: float) -> dict:
    """Describe the exact finite-sample quantile used for calibration."""
    values = np.asarray(scores, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("At least one finite calibration score is required.")
    rank = min(len(values), math.ceil((len(values) + 1) * (1 - alpha)))
    return {
        "alpha": alpha,
        "nominal_coverage": 1 - alpha,
        "n_calibration_rows": len(values),
        "finite_sample_rank": rank,
        "quantile_level": rank / len(values),
        "q_hat": finite_sample_conformal_quantile(values, alpha),
    }


def symmetric_interval(prediction, half_width) -> tuple[np.ndarray, np.ndarray]:
    """Create lower and upper bounds around a point prediction."""
    prediction = np.asarray(prediction, dtype=float)
    half_width = np.asarray(half_width, dtype=float)
    if np.any(half_width < 0):
        raise ValueError("Interval half-width cannot be negative.")
    return prediction - half_width, prediction + half_width


def interval_nonconformity_scores(y_true, lower, upper) -> np.ndarray:
    """Return the amount by which each target lies outside a base interval."""
    actual = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if not (len(actual) == len(lower) == len(upper)):
        raise ValueError("Targets and interval bounds must have the same length.")
    if np.any(lower > upper):
        raise ValueError("Lower interval bounds cannot exceed upper bounds.")
    return np.maximum.reduce([lower - actual, actual - upper, np.zeros(len(actual))])


def conformalize_interval_bounds(
    y_calibration,
    calibration_lower,
    calibration_upper,
    evaluation_lower,
    evaluation_upper,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Expand base intervals using calibration-only nonconformity scores."""
    scores = interval_nonconformity_scores(
        y_calibration,
        calibration_lower,
        calibration_upper,
    )
    q_hat = finite_sample_conformal_quantile(scores, alpha)
    lower = np.asarray(evaluation_lower, dtype=float) - q_hat
    upper = np.asarray(evaluation_upper, dtype=float) + q_hat
    return lower, upper, q_hat


def winkler_scores(y_true, lower, upper, alpha: float) -> np.ndarray:
    """Calculate row-level Winkler interval scores."""
    actual = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    width = upper - lower
    if np.any(width < 0):
        raise ValueError("Lower interval bounds cannot exceed upper bounds.")
    score = width.copy()
    below = actual < lower
    above = actual > upper
    score[below] += (2 / alpha) * (lower[below] - actual[below])
    score[above] += (2 / alpha) * (actual[above] - upper[above])
    return score


def make_interval_prediction_frame(
    y_true,
    prediction,
    lower,
    upper,
    method: str,
    model_name: str,
    evaluation_scheme: str,
    split: str,
    nominal_coverage: float,
    uncertainty_score=None,
    epsilon: float = 1e-8,
    **extra,
) -> pd.DataFrame:
    """Build one row-level prediction-interval table."""
    actual = np.asarray(y_true, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if not (len(actual) == len(prediction) == len(lower) == len(upper)):
        raise ValueError("Prediction interval arrays must have matching lengths.")
    if np.any(lower > upper):
        raise ValueError("Lower interval bounds cannot exceed upper bounds.")

    residual = actual - prediction
    absolute_error = np.abs(residual)
    width = upper - lower
    half_width = width / 2
    covered = (actual >= lower) & (actual <= upper)
    miss_direction = np.where(
        actual < lower,
        "below_interval_overprediction",
        np.where(actual > upper, "above_interval_underprediction", "covered"),
    )
    if uncertainty_score is None:
        uncertainty_score = half_width
    uncertainty_score = np.asarray(uncertainty_score, dtype=float)

    frame = pd.DataFrame(
        {
            "method": method,
            "model": model_name,
            "evaluation_scheme": evaluation_scheme,
            "split": split,
            "nominal_coverage": nominal_coverage,
            "row_position": np.arange(len(actual)),
            "Actual": actual,
            "Predicted": prediction,
            "Lower": lower,
            "Upper": upper,
            "Residual": residual,
            "AbsoluteError": absolute_error,
            "SquaredError": residual**2,
            "IntervalWidth": width,
            "HalfWidth": half_width,
            "UncertaintyScore": uncertainty_score,
            "Covered": covered,
            "MissDirection": miss_direction,
            "StandardizedError": absolute_error / np.maximum(half_width, epsilon),
            "WinklerScore": winkler_scores(
                actual,
                lower,
                upper,
                alpha=1 - nominal_coverage,
            ),
        }
    )
    for key, value in extra.items():
        frame[key] = value
    return frame


def interval_metrics(prediction_df: pd.DataFrame) -> dict:
    """Calculate point and interval metrics for one prediction table."""
    if prediction_df.empty:
        raise ValueError("Interval metrics require at least one prediction row.")
    row = prediction_df.iloc[0]
    point = regression_metrics(prediction_df["Actual"], prediction_df["Predicted"])
    coverage = float(prediction_df["Covered"].mean())
    nominal = float(row["nominal_coverage"])
    width_range = (
        prediction_df["IntervalWidth"].max()
        - prediction_df["IntervalWidth"].min()
    )
    width_tolerance = max(
        1e-8,
        abs(float(prediction_df["IntervalWidth"].median())) * 1e-8,
    )
    varying_width = width_range > width_tolerance
    uncertainty_error_spearman = (
        float(
            prediction_df["IntervalWidth"].corr(
                prediction_df["AbsoluteError"],
                method="spearman",
            )
        )
        if varying_width
        else np.nan
    )
    return {
        "method": row["method"],
        "model": row["model"],
        "evaluation_scheme": row["evaluation_scheme"],
        "split": row["split"],
        "nominal_coverage": nominal,
        "EmpiricalCoverage": coverage,
        "CoverageGap": coverage - nominal,
        "MeanIntervalWidth": float(prediction_df["IntervalWidth"].mean()),
        "MedianIntervalWidth": float(prediction_df["IntervalWidth"].median()),
        "StdIntervalWidth": float(prediction_df["IntervalWidth"].std(ddof=0)),
        "BelowIntervalRate": float(
            prediction_df["MissDirection"]
            .eq("below_interval_overprediction")
            .mean()
        ),
        "AboveIntervalRate": float(
            prediction_df["MissDirection"]
            .eq("above_interval_underprediction")
            .mean()
        ),
        "MeanWinklerScore": float(prediction_df["WinklerScore"].mean()),
        "MedianWinklerScore": float(prediction_df["WinklerScore"].median()),
        "UncertaintyErrorSpearman": uncertainty_error_spearman,
        **point,
    }


def publication_interval_metrics(prediction_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate interval metrics independently for every publication and method."""
    required = {"publication_group", "method", "evaluation_scheme"}
    missing = required - set(prediction_df.columns)
    if missing:
        raise ValueError(f"Publication interval metrics missing columns: {missing}")

    rows = []
    group_columns = ["evaluation_scheme", "method", "publication_group"]
    if "evaluation_fold" in prediction_df:
        group_columns.insert(1, "evaluation_fold")
    for keys, group in prediction_df.groupby(group_columns, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        metrics = interval_metrics(group)
        for column, value in zip(group_columns, keys):
            metrics[column] = value
        lineage_row = group.iloc[0]
        for column in (
            "publication_source",
            "publication_country",
            "publication_year",
        ):
            if column in group:
                metrics[column] = lineage_row[column]
        rows.append(metrics)
    return pd.DataFrame(rows)


def summarize_micro_macro(prediction_df: pd.DataFrame) -> pd.DataFrame:
    """Report row-weighted micro and publication-weighted macro interval metrics."""
    publication_metrics = publication_interval_metrics(prediction_df)
    rows = []
    for (scheme, method), group in prediction_df.groupby(
        ["evaluation_scheme", "method"],
        sort=False,
    ):
        micro = interval_metrics(group)
        micro["aggregation"] = "micro_all_rows"
        micro["n_publications"] = int(group["publication_group"].nunique())
        rows.append(micro)

        publication_group = publication_metrics.loc[
            publication_metrics["evaluation_scheme"].eq(scheme)
            & publication_metrics["method"].eq(method)
        ]
        macro_columns = [
            "EmpiricalCoverage",
            "CoverageGap",
            "MeanIntervalWidth",
            "MedianIntervalWidth",
            "StdIntervalWidth",
            "BelowIntervalRate",
            "AboveIntervalRate",
            "MeanWinklerScore",
            "MedianWinklerScore",
            "MAE",
            "RMSE",
            "R2",
            "R",
            "Bias",
            "MedianAE",
        ]
        macro = {
            "method": method,
            "model": publication_group["model"].iloc[0],
            "evaluation_scheme": scheme,
            "split": publication_group["split"].iloc[0],
            "nominal_coverage": publication_group["nominal_coverage"].iloc[0],
            "aggregation": "macro_equal_publication_weight",
            "n_rows": len(group),
            "n_publications": len(publication_group),
        }
        macro.update(
            {
                column: float(publication_group[column].mean())
                for column in macro_columns
                if column in publication_group
            }
        )
        macro["UncertaintyErrorSpearman"] = np.nan
        rows.append(macro)
    return pd.DataFrame(rows)


def classify_publication_confidence(
    publication_metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Label publication-level calibration behavior using coverage and width."""
    frame = publication_metrics_df.copy()
    frame["MethodMedianPublicationWidth"] = frame.groupby("method")[
        "MeanIntervalWidth"
    ].transform("median")
    below_nominal = frame["EmpiricalCoverage"] < frame["nominal_coverage"]
    width_tolerance = np.maximum(
        1e-8,
        frame["MethodMedianPublicationWidth"].abs() * 1e-8,
    )
    wide = (
        frame["MeanIntervalWidth"] - frame["MethodMedianPublicationWidth"]
    ) > width_tolerance
    frame["ConfidenceDiagnostic"] = np.select(
        [
            below_nominal & ~wide,
            below_nominal & wide,
            ~below_nominal & wide,
        ],
        ["narrow_but_wrong", "wide_and_wrong", "wide_but_safe"],
        default="well_calibrated_or_sharp",
    )
    frame["MiscoverageShortfall"] = np.maximum(
        0,
        frame["nominal_coverage"] - frame["EmpiricalCoverage"],
    )
    frame["NarrowWrongScore"] = frame["MiscoverageShortfall"] / np.maximum(
        frame["MeanIntervalWidth"],
        1e-8,
    )
    frame["CoverageEvidence"] = np.where(
        frame["n_rows"].ge(10),
        "more_stable_n_ge_10",
        "limited_n_lt_10",
    )
    return frame.sort_values(
        ["MiscoverageShortfall", "MeanIntervalWidth"],
        ascending=[False, True],
    )
