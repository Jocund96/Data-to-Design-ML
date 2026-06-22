"""Week 8 publication-generalization model experiments."""

from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from s1_linear.week07_metrics import (
    make_metrics_row,
    make_predictions_frame,
    regression_metrics,
)
from s1_linear.week07_models import build_week07_models
from s1_linear.week07_tuning import tune_week07_models


def load_split(split_dir: Path, split_name: str, target_col: str):
    """Load one aligned Week 8 predictor, target, and lineage split."""
    X = pd.read_csv(split_dir / f"X_{split_name}.csv")
    y = pd.read_csv(split_dir / f"y_{split_name}.csv")[target_col]
    lineage = pd.read_csv(split_dir / f"lineage_{split_name}.csv")
    if not (len(X) == len(y) == len(lineage)):
        raise ValueError(f"Week 8 {split_name} files are not row-aligned.")
    return X, y, lineage


def _evaluate_train_validation(
    models: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    policy: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate tuned candidates without exposing the final test split."""
    metric_rows = []
    prediction_frames = []

    for model_name, model in models.items():
        for split_name, X, y in [
            ("train", X_train, y_train),
            ("validation", X_validation, y_validation),
        ]:
            predictions = model.predict(X)
            metric_rows.append(
                make_metrics_row(
                    model_name=model_name,
                    y_true=y,
                    y_pred=predictions,
                    split=split_name,
                    policy=policy,
                    experiment="publication_held_out_model_selection",
                )
            )
            prediction_frames.append(
                make_predictions_frame(
                    model_name=model_name,
                    y_true=y,
                    y_pred=predictions,
                    split=split_name,
                    policy=policy,
                    experiment="publication_held_out_model_selection",
                )
            )
    return pd.DataFrame(metric_rows), pd.concat(prediction_frames, ignore_index=True)


def _fit_selected_model(
    model_name: str,
    frozen_config: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
):
    """Fit only the selected model using frozen hyperparameters."""
    selected_config = deepcopy(frozen_config)
    selected_config["enabled_models"] = [model_name]
    model = build_week07_models(
        X_train=X_train,
        config=selected_config,
    )[model_name]
    return model.fit(X_train, y_train), selected_config


def _prediction_frame_with_lineage(
    model_name: str,
    y_true: pd.Series,
    y_pred,
    split: str,
    policy: str,
    experiment: str,
    lineage: pd.DataFrame,
) -> pd.DataFrame:
    """Create predictions and append non-predictive lineage for interpretation."""
    predictions = make_predictions_frame(
        model_name=model_name,
        y_true=y_true,
        y_pred=y_pred,
        split=split,
        policy=policy,
        experiment=experiment,
    )
    return pd.concat(
        [predictions.reset_index(drop=True), lineage.reset_index(drop=True)],
        axis=1,
    )


def run_publication_model_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    train_lineage: pd.DataFrame,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    validation_lineage: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_lineage: pd.DataFrame,
    tuning_config: dict,
    policy: str,
    models_dir: Path,
) -> dict[str, object]:
    """
    Tune on training publications, select on validation, and test once.

    The selected model is refitted on train plus validation publications before
    the single final evaluation on unseen test publications.
    """
    tuned_models, tuning_summary, cv_results, frozen_config = tune_week07_models(
        X_train=X_train,
        y_train=y_train,
        config=tuning_config,
        groups=train_lineage["publication_group"],
        grouping_name="publication_group",
    )
    validation_metrics, validation_predictions = _evaluate_train_validation(
        models=tuned_models,
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        policy=policy,
    )
    validation_rows = validation_metrics.query("split == 'validation'").sort_values(
        "RMSE"
    )
    selected_model_name = validation_rows.iloc[0]["model"]

    X_train_validation = pd.concat(
        [X_train, X_validation],
        ignore_index=True,
    )
    y_train_validation = pd.concat(
        [y_train, y_validation],
        ignore_index=True,
    )
    selected_model, selected_config = _fit_selected_model(
        model_name=selected_model_name,
        frozen_config=frozen_config,
        X_train=X_train_validation,
        y_train=y_train_validation,
    )
    test_predictions_array = selected_model.predict(X_test)
    final_test_metrics = pd.DataFrame(
        [
            make_metrics_row(
                model_name=selected_model_name,
                y_true=y_test,
                y_pred=test_predictions_array,
                split="test",
                policy=policy,
                experiment="final_unseen_publication_test",
                training_rows=len(X_train_validation),
                training_publications=int(
                    train_lineage["publication_group"].nunique()
                    + validation_lineage["publication_group"].nunique()
                ),
            )
        ]
    )
    final_test_predictions = _prediction_frame_with_lineage(
        model_name=selected_model_name,
        y_true=y_test,
        y_pred=test_predictions_array,
        split="test",
        policy=policy,
        experiment="final_unseen_publication_test",
        lineage=test_lineage,
    )
    selected_validation = validation_rows.iloc[0]
    selected_summary = pd.DataFrame(
        [
            {
                "selection_rule": "lowest_publication_validation_RMSE",
                "selected_model": selected_model_name,
                "validation_RMSE": selected_validation["RMSE"],
                "validation_MAE": selected_validation["MAE"],
                "validation_R2": selected_validation["R2"],
                "final_test_RMSE": final_test_metrics.iloc[0]["RMSE"],
                "final_test_MAE": final_test_metrics.iloc[0]["MAE"],
                "final_test_R2": final_test_metrics.iloc[0]["R2"],
            }
        ]
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    safe_name = selected_model_name.casefold().replace(" ", "_")
    joblib.dump(selected_model, models_dir / f"week08_selected_{safe_name}.joblib")
    with (models_dir / "week08_frozen_publication_config.yaml").open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(selected_config, file, sort_keys=False)

    return {
        "tuning_summary": tuning_summary,
        "cv_results": cv_results,
        "validation_metrics": validation_metrics,
        "validation_predictions": validation_predictions,
        "selected_summary": selected_summary,
        "selected_model_name": selected_model_name,
        "selected_config": selected_config,
        "final_test_metrics": final_test_metrics,
        "final_test_predictions": final_test_predictions,
    }


def run_row_mixed_comparison(
    selected_model_name: str,
    selected_config: dict,
    row_mixed_split_dir: Path,
    publication_test_metrics: pd.DataFrame,
    target_col: str,
    policy: str,
) -> pd.DataFrame:
    """Compare the same frozen model on row-mixed and publication-held-out tests."""
    X_train = pd.read_csv(row_mixed_split_dir / "X_train.csv")
    X_validation = pd.read_csv(row_mixed_split_dir / "X_val.csv")
    X_test = pd.read_csv(row_mixed_split_dir / "X_test.csv")
    y_train = pd.read_csv(row_mixed_split_dir / "y_train.csv")[target_col]
    y_validation = pd.read_csv(row_mixed_split_dir / "y_val.csv")[target_col]
    y_test = pd.read_csv(row_mixed_split_dir / "y_test.csv")[target_col]

    X_fit = pd.concat([X_train, X_validation], ignore_index=True)
    y_fit = pd.concat([y_train, y_validation], ignore_index=True)
    model, _ = _fit_selected_model(
        model_name=selected_model_name,
        frozen_config=selected_config,
        X_train=X_fit,
        y_train=y_fit,
    )
    row_metrics = make_metrics_row(
        model_name=selected_model_name,
        y_true=y_test,
        y_pred=model.predict(X_test),
        split="test",
        policy=policy,
        experiment="row_mixed_test_same_frozen_model",
    )
    publication_metrics = publication_test_metrics.iloc[0].to_dict()
    comparison = pd.DataFrame(
        [
            {"split_strategy": "row_mixed_feature_hash", **row_metrics},
            {"split_strategy": "publication_held_out", **publication_metrics},
        ]
    )
    row = comparison.iloc[0]
    publication = comparison.iloc[1]
    comparison["RMSE_gap_vs_row_mixed"] = comparison["RMSE"] - row["RMSE"]
    comparison["MAE_gap_vs_row_mixed"] = comparison["MAE"] - row["MAE"]
    comparison["R2_gap_vs_row_mixed"] = comparison["R2"] - row["R2"]
    return comparison


def _held_out_shift_diagnostics(
    X_train: pd.DataFrame,
    X_holdout: pd.DataFrame,
) -> dict[str, float]:
    """Measure numeric-range and unseen-category shift for one held publication."""
    numeric_columns = X_train.select_dtypes(include=["number"]).columns
    categorical_columns = X_train.select_dtypes(exclude=["number"]).columns

    numeric_flags = []
    for column in numeric_columns:
        train_values = pd.to_numeric(X_train[column], errors="coerce")
        held_values = pd.to_numeric(X_holdout[column], errors="coerce")
        known = held_values.notna() & train_values.notna().any()
        if known.any():
            numeric_flags.extend(
                (
                    (held_values.loc[known] < train_values.min())
                    | (held_values.loc[known] > train_values.max())
                ).tolist()
            )

    category_flags = []
    for column in categorical_columns:
        train_categories = set(X_train[column].dropna().astype(str))
        held_values = X_holdout[column].dropna().astype(str)
        category_flags.extend((~held_values.isin(train_categories)).tolist())

    return {
        "numeric_out_of_training_range_rate": (
            float(np.mean(numeric_flags)) if numeric_flags else np.nan
        ),
        "unseen_category_rate": (
            float(np.mean(category_flags)) if category_flags else np.nan
        ),
    }


def run_leave_one_publication_out(
    modeling_df: pd.DataFrame,
    lineage_df: pd.DataFrame,
    target_col: str,
    selected_model_name: str,
    selected_config: dict,
    policy: str,
    minimum_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate the frozen selected model on every eligible held publication."""
    X = modeling_df.drop(columns=target_col)
    y = modeling_df[target_col]
    publication_counts = lineage_df["publication_group"].value_counts()
    eligible = publication_counts[publication_counts >= minimum_rows].index.tolist()
    metric_rows = []
    prediction_frames = []

    for publication_group in eligible:
        holdout_mask = lineage_df["publication_group"].eq(publication_group)
        X_train = X.loc[~holdout_mask].reset_index(drop=True)
        y_train = y.loc[~holdout_mask].reset_index(drop=True)
        X_holdout = X.loc[holdout_mask].reset_index(drop=True)
        y_holdout = y.loc[holdout_mask].reset_index(drop=True)
        holdout_lineage = lineage_df.loc[holdout_mask].reset_index(drop=True)

        model, _ = _fit_selected_model(
            model_name=selected_model_name,
            frozen_config=selected_config,
            X_train=X_train,
            y_train=y_train,
        )
        y_pred = model.predict(X_holdout)
        metrics = regression_metrics(y_holdout, y_pred)
        residual = y_holdout.to_numpy(dtype=float) - np.asarray(y_pred, dtype=float)
        squared_error = residual**2
        source_row = holdout_lineage.iloc[0]
        metric_rows.append(
            {
                "publication_group": publication_group,
                "publication_source": source_row["publication_source"],
                "publication_country": source_row["publication_country"],
                "publication_year": source_row["publication_year"],
                **metrics,
                "MaximumAE": float(np.max(np.abs(residual))),
                "target_mean": float(y_holdout.mean()),
                "target_min": float(y_holdout.min()),
                "target_max": float(y_holdout.max()),
                "worst_row_squared_error_share": float(
                    squared_error.max() / squared_error.sum()
                )
                if squared_error.sum() > 0
                else 0.0,
                "dominant_residual_direction": (
                    "underprediction"
                    if residual.mean() > 0
                    else "overprediction"
                ),
                **_held_out_shift_diagnostics(X_train, X_holdout),
            }
        )
        prediction_frames.append(
            _prediction_frame_with_lineage(
                model_name=selected_model_name,
                y_true=y_holdout,
                y_pred=y_pred,
                split="held_out_publication",
                policy=policy,
                experiment="leave_one_publication_out",
                lineage=holdout_lineage,
            )
        )

    metrics_df = pd.DataFrame(metric_rows).sort_values("RMSE", ascending=False)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    micro = regression_metrics(predictions_df["Actual"], predictions_df["Predicted"])
    macro_columns = ["MAE", "RMSE", "R2", "Bias", "MedianAE", "MaximumAE"]
    summary = pd.DataFrame(
        [
            {
                "aggregation": "micro_all_held_out_rows",
                "n_publications": len(metrics_df),
                "n_rows": len(predictions_df),
                **micro,
                "MaximumAE": predictions_df["AbsoluteError"].max(),
            },
            {
                "aggregation": "macro_equal_publication_weight",
                "n_publications": len(metrics_df),
                "n_rows": len(predictions_df),
                **{
                    column: metrics_df[column].mean()
                    for column in macro_columns
                    if column in metrics_df
                },
            },
        ]
    )
    return metrics_df, predictions_df, summary
