"""
Run Week 7 Linear Family baselines and targeted UHPC experiments.

This runner uses the agreed semantic-recoded 50% policy only. Model parameters
are fixed in config; no hyperparameter tuning is performed this week.
"""

from pathlib import Path
import argparse
import sys

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

for path in (project_root / "src", project_root, project_root.parent):
    sys.path.insert(0, str(path))

import joblib
import pandas as pd

from s1_linear.config import load_config
from s1_linear.week07_experiments import (
    run_engineering_features_experiment,
    run_error_analysis_by_groups,
    run_fiber_ablation_experiment,
    run_outlier_sensitivity_experiment,
)
from s1_linear.week07_metrics import evaluate_fitted_model
from s1_linear.week07_models import build_week07_models
from s1_linear.week07_plots import (
    plot_coefficient_importance,
    plot_error_by_group,
    plot_error_distribution,
    plot_experiment_comparison,
    plot_metrics_comparison,
    plot_predicted_vs_actual,
    plot_r2_comparison,
    plot_residuals,
    plot_top_vif,
)
from s1_linear.week07_vif import calculate_policy_vif, summarize_vif


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a config path relative to S1_Linear."""
    path = Path(path)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def load_week07_splits(split_dir: Path, target_col: str):
    """Load the fixed raw Week 7 splits used by every experiment."""
    required_files = [
        "X_train.csv",
        "X_val.csv",
        "X_test.csv",
        "y_train.csv",
        "y_val.csv",
        "y_test.csv",
    ]
    missing = [name for name in required_files if not (split_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing Week 7 split files: {missing}\n"
            "Run scripts/run_week07_preprocess_teammate_uhpc.py first."
        )

    X_train = pd.read_csv(split_dir / "X_train.csv")
    X_val = pd.read_csv(split_dir / "X_val.csv")
    X_test = pd.read_csv(split_dir / "X_test.csv")
    y_train = pd.read_csv(split_dir / "y_train.csv")[target_col]
    y_val = pd.read_csv(split_dir / "y_val.csv")[target_col]
    y_test = pd.read_csv(split_dir / "y_test.csv")[target_col]
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_baseline_models(
    policy,
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    config,
    models_dir,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train and save fixed-parameter baseline Linear Family models."""
    numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=["number"]).columns.tolist()
    models = build_week07_models(
        X_train=X_train,
        config=config,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    metric_frames = []
    prediction_frames = []
    models_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model in models.items():
        fitted = model.fit(X_train, y_train)
        metrics, predictions = evaluate_fitted_model(
            model_name=model_name,
            model=fitted,
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            policy=policy,
            experiment="main_baseline",
            experiment_family="main_baseline",
        )
        metric_frames.append(metrics)
        prediction_frames.append(predictions)

        safe_name = model_name.casefold().replace(" ", "_")
        joblib.dump(fitted, models_dir / f"week07_{safe_name}.joblib")

    return (
        pd.concat(metric_frames, ignore_index=True),
        pd.concat(prediction_frames, ignore_index=True),
    )


def select_best_baseline(
    baseline_metrics: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select the best baseline by validation RMSE and return test predictions."""
    validation = baseline_metrics.query("split == 'validation'").sort_values("RMSE")
    if validation.empty:
        raise ValueError("No validation metrics are available for model selection.")

    best_validation = validation.iloc[0]
    best_model = best_validation["model"]
    test_metric = baseline_metrics.query(
        "split == 'test' and model == @best_model"
    ).iloc[0]
    summary = pd.DataFrame(
        [
            {
                "selection_rule": "lowest_validation_RMSE",
                "best_model": best_model,
                "validation_RMSE": best_validation["RMSE"],
                "validation_MAE": best_validation["MAE"],
                "validation_R2": best_validation["R2"],
                "test_RMSE": test_metric["RMSE"],
                "test_MAE": test_metric["MAE"],
                "test_R2": test_metric["R2"],
            }
        ]
    )
    best_test_predictions = baseline_predictions.query(
        "split == 'test' and model == @best_model"
    ).reset_index(drop=True)
    return summary, best_test_predictions


def make_experiment_scope(config: dict) -> pd.DataFrame:
    """Document which Week 7 questions are run and why."""
    experiment_config = config.get("experiments", {})
    rows = []
    for experiment_name, values in experiment_config.items():
        rows.append(
            {
                "experiment": experiment_name,
                "enabled": values.get("enabled", False),
                "targeted_experiment": experiment_name
                in {"fiber_ablation", "outliers", "curing_regime_error_analysis"},
                "exploratory": values.get("exploratory", False),
                "reason": values.get("reason", ""),
            }
        )
    rows.append(
        {
            "experiment": "numeric_vif",
            "enabled": True,
            "targeted_experiment": False,
            "exploratory": False,
            "reason": "Supporting OLS multicollinearity diagnostic, not performance metric.",
        }
    )
    return pd.DataFrame(rows)


def save_table_copies(df: pd.DataFrame, table_path: Path, result_path: Path) -> None:
    """Save a report table and a results copy."""
    df.to_csv(table_path, index=False)
    df.to_csv(result_path, index=False)


def generate_week07_plots(
    output_config: dict,
    figures_dir: Path,
    models_dir: Path,
    baseline_metrics: pd.DataFrame,
    best_model_summary: pd.DataFrame,
    best_test_predictions: pd.DataFrame,
    strength_error: pd.DataFrame,
    fiber_error: pd.DataFrame,
    curing_error: pd.DataFrame,
    experiment_metrics: pd.DataFrame,
    vif_table: pd.DataFrame,
) -> None:
    """Generate the Week 7 interpretation figures from saved-result tables."""
    best_model = best_model_summary.iloc[0]["best_model"]
    best_model_path = models_dir / (
        f"week07_{best_model.casefold().replace(' ', '_')}.joblib"
    )

    plot_metrics_comparison(
        baseline_metrics,
        figures_dir / output_config["metrics_comparison_figure"],
    )
    plot_r2_comparison(
        baseline_metrics,
        figures_dir / output_config["r2_comparison_figure"],
    )
    plot_predicted_vs_actual(
        best_test_predictions,
        figures_dir / output_config["predicted_actual_figure"],
        title=f"{best_model}: Test Predictions vs Actual Strength",
    )
    plot_residuals(
        best_test_predictions,
        figures_dir / output_config["residual_figure"],
        title=f"{best_model}: Test Residuals",
    )
    plot_error_distribution(
        best_test_predictions,
        figures_dir / output_config["error_distribution_figure"],
        title=f"{best_model}: Test Absolute-Error Distribution",
    )
    plot_error_by_group(
        strength_error,
        "strength_bin",
        figures_dir / output_config["strength_error_figure"],
        title=f"{best_model}: Test MAE by Strength Range",
    )
    plot_error_by_group(
        fiber_error,
        "fiber_group",
        figures_dir / output_config["fiber_error_figure"],
        title=f"{best_model}: Test MAE by Fiber Group",
    )
    plot_error_by_group(
        curing_error,
        "curing_group",
        figures_dir / output_config["curing_error_figure"],
        title=f"{best_model}: Test MAE by Curing Group",
    )
    plot_top_vif(
        vif_table,
        figures_dir / output_config["vif_figure"],
    )

    experiment_figures = {
        "fiber_ablation": "fiber_ablation_figure",
        "outlier_sensitivity": "outlier_sensitivity_figure",
        "engineering_features": "engineering_features_figure",
    }
    for experiment_family, figure_key in experiment_figures.items():
        plot_experiment_comparison(
            experiment_metrics,
            figures_dir / output_config[figure_key],
            experiment_family,
        )

    fitted_pipeline = joblib.load(best_model_path)
    plot_coefficient_importance(
        fitted_pipeline,
        figures_dir / output_config["coefficient_figure"],
    )


def main(config_path: str) -> None:
    """Run the complete Week 7 baseline and targeted-experiment workflow."""
    config = load_config(resolve_project_path(config_path))
    split_dir = resolve_project_path(config["data"]["split_dir"])
    target_col = config["data"]["target"]
    policy = config["data"]["policy"]
    output_config = config["outputs"]

    tables_dir = resolve_project_path(output_config["tables_dir"])
    metrics_dir = resolve_project_path(output_config["metrics_dir"])
    predictions_dir = resolve_project_path(output_config["predictions_dir"])
    models_dir = resolve_project_path(output_config["models_dir"])
    figures_dir = resolve_project_path(output_config["figures_dir"])
    for directory in [
        tables_dir,
        metrics_dir,
        predictions_dir,
        models_dir,
        figures_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    X_train, X_val, X_test, y_train, y_val, y_test = load_week07_splits(
        split_dir=split_dir,
        target_col=target_col,
    )
    numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=["number"]).columns.tolist()

    baseline_metrics, baseline_predictions = train_baseline_models(
        policy=policy,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        config=config,
        models_dir=models_dir,
    )
    save_table_copies(
        baseline_metrics,
        tables_dir / output_config["baseline_metrics_name"],
        metrics_dir / output_config["baseline_metrics_name"],
    )
    save_table_copies(
        baseline_predictions,
        tables_dir / output_config["baseline_predictions_name"],
        predictions_dir / output_config["baseline_predictions_name"],
    )

    best_model_summary, best_test_predictions = select_best_baseline(
        baseline_metrics,
        baseline_predictions,
    )
    best_model_summary.to_csv(
        tables_dir / output_config["best_model_name"],
        index=False,
    )

    strength_error, fiber_error, curing_error = run_error_analysis_by_groups(
        best_prediction_df=best_test_predictions,
        X_test=X_test,
    )
    strength_error.to_csv(
        tables_dir / output_config["strength_error_name"],
        index=False,
    )
    fiber_error.to_csv(tables_dir / output_config["fiber_error_name"], index=False)
    curing_error.to_csv(tables_dir / output_config["curing_error_name"], index=False)

    experiment_metric_frames = []
    experiment_prediction_frames = []
    experiment_config = config.get("experiments", {})

    if experiment_config.get("fiber_ablation", {}).get("enabled", True):
        metrics, predictions = run_fiber_ablation_experiment(
            policy,
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            config,
            numeric_features,
            categorical_features,
        )
        experiment_metric_frames.append(metrics)
        experiment_prediction_frames.append(predictions)

    if experiment_config.get("outliers", {}).get("enabled", True):
        metrics, predictions = run_outlier_sensitivity_experiment(
            policy,
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            config,
            numeric_features,
            categorical_features,
        )
        experiment_metric_frames.append(metrics)
        experiment_prediction_frames.append(predictions)

    if experiment_config.get("engineering_features", {}).get("enabled", True):
        metrics, predictions = run_engineering_features_experiment(
            policy,
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            config,
            numeric_features,
            categorical_features,
        )
        experiment_metric_frames.append(metrics)
        experiment_prediction_frames.append(predictions)

    experiment_metrics = pd.concat(experiment_metric_frames, ignore_index=True)
    experiment_predictions = pd.concat(
        experiment_prediction_frames,
        ignore_index=True,
    )
    save_table_copies(
        experiment_metrics,
        tables_dir / output_config["experiment_metrics_name"],
        metrics_dir / output_config["experiment_metrics_name"],
    )
    save_table_copies(
        experiment_predictions,
        tables_dir / output_config["experiment_predictions_name"],
        predictions_dir / output_config["experiment_predictions_name"],
    )

    vif_table, vif_audit = calculate_policy_vif(
        X_train=X_train,
        numeric_features=numeric_features,
        policy=policy,
    )
    vif_summary = summarize_vif(vif_table)
    vif_table.to_csv(tables_dir / output_config["vif_name"], index=False)
    vif_audit.to_csv(tables_dir / output_config["vif_audit_name"], index=False)
    vif_summary.to_csv(tables_dir / output_config["vif_summary_name"], index=False)

    scope = make_experiment_scope(config)
    scope.to_csv(tables_dir / output_config["experiment_scope_name"], index=False)

    generate_week07_plots(
        output_config=output_config,
        figures_dir=figures_dir,
        models_dir=models_dir,
        baseline_metrics=baseline_metrics,
        best_model_summary=best_model_summary,
        best_test_predictions=best_test_predictions,
        strength_error=strength_error,
        fiber_error=fiber_error,
        curing_error=curing_error,
        experiment_metrics=experiment_metrics,
        vif_table=vif_table,
    )

    print("\nWeek 7 baseline validation metrics:")
    print(
        baseline_metrics.query("split == 'validation'")
        .sort_values("RMSE")
        .round(3)
        .to_string(index=False)
    )
    print("\nSelected baseline:")
    print(best_model_summary.round(3).to_string(index=False))
    print("\nVIF summary:")
    print(vif_summary.round(3).to_string(index=False))
    print("\nWeek 7 targeted experiments complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Week 7 Linear Family baselines and targeted experiments."
    )
    parser.add_argument(
        "--config",
        default="configs/week07_linear_experiments.yaml",
        help="Config path relative to S1_Linear by default.",
    )
    args = parser.parse_args()
    main(args.config)
