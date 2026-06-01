"""
Week 4 - Representation Experiments for Linear Family models.

This script runs different feature representation experiments:
1. Original features
2. Yeh/domain engineered features
3. Log-transformed features
4. Aggressive interaction features

For each representation, it trains the Linear Family models:
- OLS
- Elastic Net
- Bayesian Ridge
- Polynomial Ridge

Then it:
- saves metrics
- saves best parameters
- saves predictions
- creates residual and actual-vs-predicted plots
- re-runs the best setup with three seeds and reports mean ± std
"""

import os
import yaml
import joblib
import pandas as pd

from sklearn.model_selection import GridSearchCV, train_test_split

from s1_linear.data import load_dataset, split_features_target
from s1_linear.features import build_feature_representation
from s1_linear.models import (
    build_ols_pipeline,
    build_elastic_net_pipeline,
    build_bayesian_ridge_pipeline,
    build_polynomial_ridge_pipeline,
)
from s1_linear.evaluation import evaluate_model
from s1_linear.plots import plot_actual_vs_predicted, plot_residuals


def load_config(path: str = "configs/week04_representation.yaml") -> dict:
    """Load YAML config file."""
    with open(path, "r") as file:
        return yaml.safe_load(file)


def make_dirs(config: dict) -> None:
    """Create output directories if they do not exist."""
    for key in [
        "figures_dir",
        "tables_dir",
        "metrics_dir",
        "predictions_dir",
        "models_dir",
    ]:
        os.makedirs(config["outputs"][key], exist_ok=True)


def train_and_evaluate_models(
    X_train,
    X_test,
    y_train,
    y_test,
    cv_folds: int,
    random_state: int,
):
    """
    Train and evaluate all Linear Family models.

    Returns:
        results: list of metric dictionaries
        best_params: list of best parameter dictionaries
        fitted_models: dictionary of trained models
    """
    results = []
    best_params = []
    fitted_models = {}

    # ---------------------------------------------------------
    # 1. OLS Linear Regression
    # ---------------------------------------------------------
    ols = build_ols_pipeline()
    ols.fit(X_train, y_train)

    results.append(evaluate_model("OLS", ols, X_test, y_test))
    best_params.append({"Model": "OLS", "BestParams": "No hyperparameters tuned"})
    fitted_models["OLS"] = ols

    # ---------------------------------------------------------
    # 2. Elastic Net
    # ---------------------------------------------------------
    elastic = build_elastic_net_pipeline(random_state=random_state)

    elastic_params = {
        "model__alpha": [0.001, 0.01, 0.1, 1, 10, 100],
        "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    }

    elastic_grid = GridSearchCV(
        elastic,
        elastic_params,
        cv=cv_folds,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )

    elastic_grid.fit(X_train, y_train)
    best_elastic = elastic_grid.best_estimator_

    results.append(evaluate_model("Elastic Net", best_elastic, X_test, y_test))
    best_params.append(
        {"Model": "Elastic Net", "BestParams": str(elastic_grid.best_params_)}
    )
    fitted_models["Elastic Net"] = best_elastic

    # ---------------------------------------------------------
    # 3. Bayesian Ridge
    # ---------------------------------------------------------
    bayesian = build_bayesian_ridge_pipeline()
    bayesian.fit(X_train, y_train)

    results.append(evaluate_model("Bayesian Ridge", bayesian, X_test, y_test))
    best_params.append(
        {"Model": "Bayesian Ridge", "BestParams": "Default sklearn parameters"}
    )
    fitted_models["Bayesian Ridge"] = bayesian

    # ---------------------------------------------------------
    # 4. Polynomial Ridge
    # ---------------------------------------------------------
    poly = build_polynomial_ridge_pipeline()

    # poly_params = {
    #     "poly__degree": [1, 2, 3],
    #     "model__alpha": [0.01, 0.1, 1, 10, 100],
    # }
    n_features = X_train.shape[1]

    if n_features <= 14:
        poly_degrees = [1, 2, 3]
    else:
        poly_degrees = [1, 2]

    poly_params = {
        "poly__degree": poly_degrees,
        "model__alpha": [0.01, 0.1, 1, 10, 100],
    }

    poly_grid = GridSearchCV(
        poly,
        poly_params,
        cv=cv_folds,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )

    poly_grid.fit(X_train, y_train)
    best_poly = poly_grid.best_estimator_

    results.append(evaluate_model("Polynomial Ridge", best_poly, X_test, y_test))
    best_params.append(
        {"Model": "Polynomial Ridge", "BestParams": str(poly_grid.best_params_)}
    )
    fitted_models["Polynomial Ridge"] = best_poly

    return results, best_params, fitted_models


def run_single_representation(
    config: dict,
    feature_set_name: str,
    random_state: int,
    save_outputs: bool = True,
):
    """
    Run all Linear Family models for one feature representation.

    Example feature_set_name:
    - original
    - yeh_engineered
    - log_transformed
    - aggressive_interactions
    """
    data_path = config["data"]["processed_path"]
    target = config["data"]["target"]
    test_size = config["split"]["test_size"]
    cv_folds = config["split"]["cv_folds"]

    feature_config = config["feature_sets"][feature_set_name]
    representation = feature_config["representation"]
    selected_features = feature_config["features"]

    df = load_dataset(data_path)

    # Build selected representation
    df = build_feature_representation(df, representation)

    missing_columns = [
        col for col in selected_features + [target] if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns for feature set '{feature_set_name}': {missing_columns}"
        )

    # Select features and target
    X, y = split_features_target(df, selected_features, target)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    results, best_params, fitted_models = train_and_evaluate_models(
        X_train,
        X_test,
        y_train,
        y_test,
        cv_folds=cv_folds,
        random_state=random_state,
    )

    # Add representation info to result rows
    for row in results:
        row["FeatureSet"] = feature_set_name
        row["RandomState"] = random_state

    for row in best_params:
        row["FeatureSet"] = feature_set_name
        row["RandomState"] = random_state

    if save_outputs:
        save_representation_outputs(
            config=config,
            feature_set_name=feature_set_name,
            random_state=random_state,
            results=results,
            best_params=best_params,
            fitted_models=fitted_models,
            X_test=X_test,
            y_test=y_test,
        )

    return results, best_params, fitted_models


def save_representation_outputs(
    config: dict,
    feature_set_name: str,
    random_state: int,
    results: list,
    best_params: list,
    fitted_models: dict,
    X_test,
    y_test,
) -> None:
    """Save metrics, parameters, models, predictions, and plots."""
    figures_dir = config["outputs"]["figures_dir"]
    tables_dir = config["outputs"]["tables_dir"]
    models_dir = config["outputs"]["models_dir"]
    predictions_dir = config["outputs"]["predictions_dir"]

    # Save metrics
    metrics_df = pd.DataFrame(results)
    metrics_path = (
        f"{tables_dir}/week04_{feature_set_name}_metrics_seed_{random_state}.csv"
    )
    metrics_df.to_csv(metrics_path, index=False)

    # Save best params
    params_df = pd.DataFrame(best_params)
    params_path = (
        f"{tables_dir}/week04_{feature_set_name}_best_params_seed_{random_state}.csv"
    )
    params_df.to_csv(params_path, index=False)

    # Save predictions and plots
    all_predictions = []

    for model_name, model in fitted_models.items():
        y_pred = model.predict(X_test)

        safe_model_name = model_name.lower().replace(" ", "_")
        safe_feature_name = feature_set_name.lower()

        # Save model
        model_path = f"{models_dir}/week04_{safe_feature_name}_{safe_model_name}_seed_{random_state}.joblib"
        joblib.dump(model, model_path)

        # Save predictions
        prediction_df = pd.DataFrame(
            {
                "FeatureSet": feature_set_name,
                "Model": model_name,
                "RandomState": random_state,
                "Actual": y_test.values,
                "Predicted": y_pred,
                "Residual": y_test.values - y_pred,
            }
        )
        all_predictions.append(prediction_df)

        # Save plots
        plot_actual_vs_predicted(
            y_test,
            y_pred,
            f"{model_name} - {feature_set_name}",
            save_path=f"{figures_dir}/week04_{safe_feature_name}_{safe_model_name}_actual_vs_predicted_seed_{random_state}.png",
        )

        plot_residuals(
            y_test,
            y_pred,
            f"{model_name} - {feature_set_name}",
            save_path=f"{figures_dir}/week04_{safe_feature_name}_{safe_model_name}_residuals_seed_{random_state}.png",
        )

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    prediction_path = f"{predictions_dir}/week04_{feature_set_name}_predictions_seed_{random_state}.csv"
    predictions_df.to_csv(prediction_path, index=False)


def run_all_representations(config: dict) -> pd.DataFrame:
    """
    Run the main Week 4 experiments using the default random seed.

    This is the first part of the Week 4 task:
    Perform at least three representation experiments.
    """
    random_state = config["split"]["random_state"]

    all_results = []
    all_params = []

    for feature_set_name in config["feature_sets"].keys():
        print(f"\nRunning feature set: {feature_set_name}")

        results, best_params, _ = run_single_representation(
            config=config,
            feature_set_name=feature_set_name,
            random_state=random_state,
            save_outputs=True,
        )

        all_results.extend(results)
        all_params.extend(best_params)

    results_df = pd.DataFrame(all_results)
    params_df = pd.DataFrame(all_params)

    results_df = results_df[
        ["FeatureSet", "Model", "RandomState", "MAE", "RMSE", "R", "R2"]
    ]

    results_df.to_csv(
        f"{config['outputs']['tables_dir']}/week04_all_representation_metrics.csv",
        index=False,
    )

    params_df.to_csv(
        f"{config['outputs']['tables_dir']}/week04_all_best_params.csv",
        index=False,
    )

    return results_df


def find_best_setup(results_df: pd.DataFrame):
    """
    Find the best setup based on the lowest RMSE.

    Returns:
        best_feature_set
        best_model
    """
    best_row = results_df.sort_values("RMSE", ascending=True).iloc[0]

    best_feature_set = best_row["FeatureSet"]
    best_model = best_row["Model"]

    return best_feature_set, best_model


def run_seed_test(
    config: dict, best_feature_set: str, best_model_name: str
) -> pd.DataFrame:
    """
    Re-running the best feature setup using three different seeds.

    I have reported mean ± std.
    Here I ran all models again for the best feature set, then filter
    the selected best model.
    """
    seeds = config["seed_test"]["seeds"]

    seed_results = []

    for seed in seeds:
        print(f"\nSeed test: feature set = {best_feature_set}, seed = {seed}")

        results, _, _ = run_single_representation(
            config=config,
            feature_set_name=best_feature_set,
            random_state=seed,
            save_outputs=False,
        )

        for row in results:
            if row["Model"] == best_model_name:
                seed_results.append(row)

    seed_df = pd.DataFrame(seed_results)

    summary = (
        seed_df.groupby(["FeatureSet", "Model"])
        .agg(
            {
                "MAE": ["mean", "std"],
                "RMSE": ["mean", "std"],
                "R": ["mean", "std"],
                "R2": ["mean", "std"],
            }
        )
        .reset_index()
    )

    # Flatten column names
    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    seed_df.to_csv(
        f"{config['outputs']['tables_dir']}/week04_seed_test_raw_results.csv",
        index=False,
    )

    summary.to_csv(
        f"{config['outputs']['tables_dir']}/week04_seed_test_mean_std.csv",
        index=False,
    )

    return summary


def main():
    config = load_config()
    make_dirs(config)

    print("\nStarting Week 4 representation experiments...")

    results_df = run_all_representations(config)

    print("\nAll representation results:")
    print(results_df.round(3))

    best_feature_set, best_model = find_best_setup(results_df)

    print("\nBest setup based on lowest RMSE:")
    print(f"Feature set: {best_feature_set}")
    print(f"Model: {best_model}")

    seed_summary = run_seed_test(
        config=config,
        best_feature_set=best_feature_set,
        best_model_name=best_model,
    )

    print("\nSeed test mean ± std:")
    print(seed_summary.round(3))

    print("\nWeek 4 experiments completed.")


if __name__ == "__main__":
    main()
