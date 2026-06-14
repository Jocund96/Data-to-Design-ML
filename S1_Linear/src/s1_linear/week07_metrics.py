"""Regression metrics and prediction tables for Week 7 experiments."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict:
    """Calculate fixed Week 7 regression metrics."""
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    residual = y_true_array - y_pred_array

    if np.std(y_true_array) == 0 or np.std(y_pred_array) == 0:
        correlation = np.nan
    else:
        correlation = float(np.corrcoef(y_true_array, y_pred_array)[0, 1])

    return {
        "n_rows": int(len(y_true_array)),
        "MAE": float(mean_absolute_error(y_true_array, y_pred_array)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true_array, y_pred_array))),
        "R2": float(r2_score(y_true_array, y_pred_array)),
        "R": correlation,
        "Bias": float(np.mean(residual)),
        "MedianAE": float(np.median(np.abs(residual))),
    }


def make_metrics_row(
    model_name: str,
    y_true,
    y_pred,
    split: str,
    policy: str,
    experiment: str,
    exploratory: bool = False,
    **extra,
) -> dict:
    """Create one labelled metrics row."""
    return {
        "policy": policy,
        "experiment": experiment,
        "exploratory": exploratory,
        "model": model_name,
        "split": split,
        **regression_metrics(y_true, y_pred),
        **extra,
    }


def make_predictions_frame(
    model_name: str,
    y_true,
    y_pred,
    split: str,
    policy: str,
    experiment: str,
    exploratory: bool = False,
    **extra,
) -> pd.DataFrame:
    """Create labelled predictions with residuals."""
    y_true_series = pd.Series(y_true).reset_index(drop=True)
    y_pred_array = np.asarray(y_pred, dtype=float)
    residual = y_true_series.to_numpy(dtype=float) - y_pred_array

    frame = pd.DataFrame(
        {
            "policy": policy,
            "experiment": experiment,
            "exploratory": exploratory,
            "model": model_name,
            "split": split,
            "row_position": np.arange(len(y_true_series)),
            "Actual": y_true_series,
            "Predicted": y_pred_array,
            "Residual": residual,
            "AbsoluteError": np.abs(residual),
            "SquaredError": residual**2,
        }
    )

    for key, value in extra.items():
        frame[key] = value

    return frame


def evaluate_fitted_model(
    model_name: str,
    model,
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    policy: str,
    experiment: str,
    exploratory: bool = False,
    **extra,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a fitted model on train, validation, and test splits."""
    split_data = [
        ("train", X_train, y_train),
        ("validation", X_val, y_val),
        ("test", X_test, y_test),
    ]
    metrics_rows = []
    prediction_frames = []

    for split_name, X_split, y_split in split_data:
        y_pred = model.predict(X_split)
        metrics_rows.append(
            make_metrics_row(
                model_name=model_name,
                y_true=y_split,
                y_pred=y_pred,
                split=split_name,
                policy=policy,
                experiment=experiment,
                exploratory=exploratory,
                **extra,
            )
        )
        prediction_frames.append(
            make_predictions_frame(
                model_name=model_name,
                y_true=y_split,
                y_pred=y_pred,
                split=split_name,
                policy=policy,
                experiment=experiment,
                exploratory=exploratory,
                **extra,
            )
        )

    return pd.DataFrame(metrics_rows), pd.concat(prediction_frames, ignore_index=True)


def summarize_prediction_errors_by_group(
    prediction_df: pd.DataFrame,
    group: pd.Series,
    group_name: str,
) -> pd.DataFrame:
    """Summarize prediction error for one grouping variable."""
    frame = prediction_df.reset_index(drop=True).copy()
    frame[group_name] = pd.Series(group).reset_index(drop=True).astype(str)

    rows = []
    for group_value, group_df in frame.groupby(group_name, dropna=False):
        rows.append(
            {
                group_name: group_value,
                "n_rows": len(group_df),
                "MAE": group_df["AbsoluteError"].mean(),
                "RMSE": np.sqrt(group_df["SquaredError"].mean()),
                "Bias": group_df["Residual"].mean(),
                "MedianAE": group_df["AbsoluteError"].median(),
                "ActualMean": group_df["Actual"].mean(),
                "PredictedMean": group_df["Predicted"].mean(),
            }
        )

    return pd.DataFrame(rows).sort_values(["RMSE", group_name], ascending=[False, True])
