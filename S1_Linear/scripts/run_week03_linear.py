from pathlib import Path
import argparse
import sys

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
for path in (project_root, project_root.parent):
    sys.path.insert(0, str(path))

import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV

from s1_linear.config import load_config
from s1_linear.data import load_dataset, split_features_target, make_train_test_split
from s1_linear.models import (
    build_ols_pipeline,
    build_elastic_net_pipeline,
    build_bayesian_ridge_pipeline,
    build_polynomial_ridge_pipeline,
)
from s1_linear.evaluation import evaluate_model, make_predictions_frame
from s1_linear.plots import (
    plot_actual_vs_predicted,
    plot_residuals,
    plot_residuals_vs_age,
)


def make_dirs(*paths):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def tune_model(model, param_grid, X_train, y_train, cv_folds: int):
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv_folds,
        n_jobs=1,
    )
    grid.fit(X_train, y_train)
    return grid


def main(config_path: str):
    config = load_config(config_path)

    data_path = config["data"]["path"]
    target = config["data"]["target"]
    features = config["features"]["original_yeh"]

    test_size = config["split"]["test_size"]
    random_state = config["split"]["random_state"]
    cv_folds = config["split"]["cv_folds"]

    tables_dir = Path(config["outputs"]["tables_dir"])
    figures_dir = Path(config["outputs"]["figures_dir"])
    metrics_dir = Path(config["outputs"]["metrics_dir"])
    predictions_dir = Path(config["outputs"]["predictions_dir"])
    models_dir = Path(config["outputs"]["models_dir"])
    make_dirs(tables_dir, figures_dir, metrics_dir, predictions_dir, models_dir)

    df = load_dataset(data_path)
    X, y = split_features_target(df, features, target)
    X_train, X_test, y_train, y_test = make_train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    results = []
    all_predictions = []
    best_params = []

    # 1. OLS baseline
    ols = build_ols_pipeline()
    ols.fit(X_train, y_train)
    results.append(evaluate_model("OLS", ols, X_test, y_test))
    best_params.append({"Model": "OLS", "Best parameters": "None"})

    # 2. Elastic Net tuned by GridSearchCV
    elastic = build_elastic_net_pipeline(random_state=random_state)
    elastic_grid = tune_model(
        elastic,
        {
            "model__alpha": config["models"]["elastic_net"]["alpha"],
            "model__l1_ratio": config["models"]["elastic_net"]["l1_ratio"],
        },
        X_train,
        y_train,
        cv_folds,
    )
    elastic_best = elastic_grid.best_estimator_
    results.append(evaluate_model("Elastic Net", elastic_best, X_test, y_test))
    best_params.append(
        {"Model": "Elastic Net", "Best parameters": str(elastic_grid.best_params_)}
    )

    # 3. Bayesian Ridge baseline without tuning (since it has no major hyperparameters to tune)
    bayesian = build_bayesian_ridge_pipeline()
    bayesian.fit(X_train, y_train)
    results.append(evaluate_model("Bayesian Ridge", bayesian, X_test, y_test))
    best_params.append(
        {"Model": "Bayesian Ridge", "Best parameters": "Default sklearn parameters"}
    )

    # 4. Polynomial Ridge tuned by GridSearchCV
    poly = build_polynomial_ridge_pipeline()
    poly_grid = tune_model(
        poly,
        {
            "poly__degree": config["models"]["polynomial_ridge"]["degree"],
            "model__alpha": config["models"]["polynomial_ridge"]["alpha"],
        },
        X_train,
        y_train,
        cv_folds,
    )
    poly_best = poly_grid.best_estimator_
    results.append(evaluate_model("Polynomial Ridge", poly_best, X_test, y_test))
    best_params.append(
        {"Model": "Polynomial Ridge", "Best parameters": str(poly_grid.best_params_)}
    )

    fitted_models = {
        "OLS": ols,
        "Elastic Net": elastic_best,
        "Bayesian Ridge": bayesian,
        "Polynomial Ridge": poly_best,
    }

    for model_name, model in fitted_models.items():
        safe_name = model_name.lower().replace(" ", "_")
        y_pred = model.predict(X_test)
        all_predictions.append(make_predictions_frame(y_test, y_pred, model_name))

        plot_actual_vs_predicted(
            y_test,
            y_pred,
            model_name,
            figures_dir / f"{safe_name}_actual_vs_predicted.png",
        )
        plot_residuals(
            y_test,
            y_pred,
            model_name,
            figures_dir / f"{safe_name}_residuals.png",
        )
        plot_residuals_vs_age(
            X_test,
            y_test,
            y_pred,
            model_name,
            figures_dir / f"{safe_name}_residuals_vs_age.png",
        )

    results_df = pd.DataFrame(results)[["Model", "MAE", "RMSE", "R", "R2"]]
    params_df = pd.DataFrame(best_params)
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    results_df.to_csv(tables_dir / "week03_linear_metrics.csv", index=False)
    results_df.to_csv(metrics_dir / "week03_linear_metrics.csv", index=False)
    params_df.to_csv(tables_dir / "week03_linear_best_params.csv", index=False)
    predictions_df.to_csv(
        predictions_dir / "week03_linear_predictions.csv", index=False
    )

    joblib.dump(ols, models_dir / "ols.joblib")
    joblib.dump(elastic_best, models_dir / "elastic_net.joblib")
    joblib.dump(bayesian, models_dir / "bayesian_ridge.joblib")
    joblib.dump(poly_best, models_dir / "polynomial_ridge.joblib")

    print("\nWeek 3 Linear Family results")
    print(results_df.round(3).to_string(index=False))
    print("\nBest parameters")
    print(params_df.to_string(index=False))
    print("\nSaved outputs to reports/ and results/.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Week 3 Linear Family baselines.")
    parser.add_argument(
        "--config",
        default="configs/week03_linear.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()
    main(args.config)
