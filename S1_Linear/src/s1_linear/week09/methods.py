"""Train-only Linear Family uncertainty methods for Week 9."""

from copy import deepcopy
from statistics import NormalDist

import numpy as np
from sklearn.base import clone

from s1_linear.week07_models import build_week07_models


def fit_frozen_pipeline(
    model_name: str,
    frozen_config: dict,
    X_train,
    y_train,
):
    """Fit one frozen Week 8 model configuration on supplied training rows only."""
    config = deepcopy(frozen_config)
    config["enabled_models"] = [model_name]
    config.setdefault("hyperparameter_tuning", {})["enabled"] = False
    models = build_week07_models(X_train=X_train, config=config)
    if model_name not in models:
        raise ValueError(f"Frozen model builder unavailable: {model_name}")
    return models[model_name].fit(X_train, y_train), config


def predict_bayesian_distribution(fitted_pipeline, X) -> tuple[np.ndarray, np.ndarray]:
    """Return Bayesian Ridge predictive mean and standard deviation."""
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    model = fitted_pipeline.named_steps["model"]
    transformed = preprocessor.transform(X)
    mean, standard_deviation = model.predict(transformed, return_std=True)
    return np.asarray(mean, dtype=float), np.asarray(standard_deviation, dtype=float)


def normal_interval(
    mean,
    standard_deviation,
    nominal_coverage: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Create a central Gaussian predictive interval."""
    if not 0 < nominal_coverage < 1:
        raise ValueError("nominal_coverage must be strictly between 0 and 1.")
    z_value = NormalDist().inv_cdf((1 + nominal_coverage) / 2)
    mean = np.asarray(mean, dtype=float)
    standard_deviation = np.asarray(standard_deviation, dtype=float)
    half_width = z_value * standard_deviation
    return mean - half_width, mean + half_width, float(z_value)


def residual_bootstrap_prediction_distributions(
    fitted_elastic_net_pipeline,
    X_train,
    y_train,
    prediction_sets: dict[str, object],
    repetitions: int,
    random_state: int,
) -> dict[str, np.ndarray]:
    """
    Generate residual-bootstrap predictive distributions.

    Preprocessing is fitted once on the training predictors. It is unsupervised
    and receives identical X values in every repetition, so reusing its fixed
    transformed matrix is equivalent to refitting it while avoiding needless
    repeated encoding and scaling.
    """
    if repetitions < 2:
        raise ValueError("Residual bootstrap requires at least two repetitions.")

    preprocessor = fitted_elastic_net_pipeline.named_steps["preprocessor"]
    base_model = fitted_elastic_net_pipeline.named_steps["model"]
    transformed_train = preprocessor.transform(X_train)
    transformed_sets = {
        name: preprocessor.transform(X) for name, X in prediction_sets.items()
    }
    y_train_array = np.asarray(y_train, dtype=float)
    fitted_train = np.asarray(
        base_model.predict(transformed_train),
        dtype=float,
    )
    centered_residuals = y_train_array - fitted_train
    centered_residuals = centered_residuals - centered_residuals.mean()
    if not np.any(np.abs(centered_residuals) > 0):
        raise ValueError("Residual bootstrap cannot use all-zero training residuals.")

    rng = np.random.default_rng(random_state)
    distributions = {
        name: np.empty((repetitions, transformed.shape[0]), dtype=float)
        for name, transformed in transformed_sets.items()
    }

    for repetition in range(repetitions):
        pseudo_target = fitted_train + rng.choice(
            centered_residuals,
            size=len(centered_residuals),
            replace=True,
        )
        bootstrap_model = clone(base_model).fit(transformed_train, pseudo_target)
        for name, transformed in transformed_sets.items():
            prediction = bootstrap_model.predict(transformed)
            predictive_noise = rng.choice(
                centered_residuals,
                size=len(prediction),
                replace=True,
            )
            distributions[name][repetition] = prediction + predictive_noise

    return distributions


def bootstrap_percentile_interval(
    prediction_distribution: np.ndarray,
    nominal_coverage: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return central percentile bounds from bootstrap predictions."""
    if prediction_distribution.ndim != 2:
        raise ValueError("Bootstrap distribution must have shape repetitions x rows.")
    alpha = 1 - nominal_coverage
    lower = np.quantile(prediction_distribution, alpha / 2, axis=0)
    upper = np.quantile(prediction_distribution, 1 - alpha / 2, axis=0)
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)

