import pandas as pd

from s1_linear.week07_models import build_week07_models
from s1_linear.week07_tuning import make_frozen_model_config


def _config():
    return {
        "enabled_models": [
            "OLS",
            "Elastic Net",
            "Bayesian Ridge",
            "Polynomial Ridge",
        ],
        "preprocessing": {},
        "models": {
            "elastic_net": {},
            "bayesian_ridge": {},
            "polynomial_ridge": {"degree": 2, "solver": "lsqr"},
        },
        "hyperparameter_tuning": {"enabled": True},
    }


def test_week07_registry_contains_only_selected_four_models():
    X_train = pd.DataFrame(
        {
            "cement": [700.0, 800.0, 900.0],
            "curing_method": ["Standard Curing", "Water Curing", "Heat Curing"],
        }
    )

    models = build_week07_models(X_train=X_train, config=_config())

    assert list(models) == [
        "OLS",
        "Elastic Net",
        "Bayesian Ridge",
        "Polynomial Ridge",
    ]
    assert list(models["Polynomial Ridge"].named_steps) == [
        "preprocessor",
        "poly",
        "polynomial_scaler",
        "model",
    ]


def test_tuned_pipeline_parameters_are_frozen_for_experiments():
    frozen = make_frozen_model_config(
        _config(),
        {
            "OLS": {},
            "Elastic Net": {"model__alpha": 0.01, "model__l1_ratio": 0.9},
            "Bayesian Ridge": {"model__alpha_1": 1e-5},
            "Polynomial Ridge": {"poly__degree": 2, "model__alpha": 100.0},
        },
    )

    assert frozen["hyperparameter_tuning"]["enabled"] is False
    assert frozen["hyperparameter_tuning"]["frozen_from_training_cv"] is True
    assert frozen["models"]["elastic_net"]["l1_ratio"] == 0.9
    assert frozen["models"]["bayesian_ridge"]["alpha_1"] == 1e-5
    assert frozen["models"]["polynomial_ridge"]["degree"] == 2
    assert frozen["models"]["polynomial_ridge"]["alpha"] == 100.0
