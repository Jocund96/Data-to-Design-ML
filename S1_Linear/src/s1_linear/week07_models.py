"""Fixed Linear Family model builders for Week 7 UHPC experiments."""

from sklearn.linear_model import BayesianRidge, ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline

from s1_linear.week07_preprocessing import build_week07_preprocessor


def _model_config(config: dict, model_key: str) -> dict:
    """Return model-specific config with a safe empty default."""
    return config.get("models", {}).get(model_key, {})


def build_week07_model_pipeline(
    model,
    X_train,
    config: dict,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> Pipeline:
    """Build one model pipeline with preprocessing fitted later on X_train only."""
    preprocessing_config = config.get("preprocessing", {})
    build = build_week07_preprocessor(
        X_train=X_train,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        numeric_add_indicator=preprocessing_config.get(
            "numeric_add_indicator", True
        ),
        categorical_missing_value=preprocessing_config.get(
            "categorical_missing_value",
            "missing_reported_gap",
        ),
        categorical_min_frequency=preprocessing_config.get(
            "categorical_min_frequency",
            5,
        ),
        categorical_max_categories=preprocessing_config.get(
            "categorical_max_categories",
            25,
        ),
    )

    return Pipeline(
        [
            ("preprocessor", build.preprocessor),
            ("model", model),
        ]
    )


def build_ols_model(
    X_train,
    config: dict,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> Pipeline:
    """Ordinary Least Squares baseline with Week 7 preprocessing."""
    return build_week07_model_pipeline(
        LinearRegression(),
        X_train=X_train,
        config=config,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )


def build_ridge_model(
    X_train,
    config: dict,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> Pipeline:
    """Ridge baseline with fixed alpha from config."""
    params = _model_config(config, "ridge")
    return build_week07_model_pipeline(
        Ridge(
            alpha=params.get("alpha", 10.0),
            random_state=params.get("random_state"),
        ),
        X_train=X_train,
        config=config,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )


def build_lasso_model(
    X_train,
    config: dict,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> Pipeline:
    """Lasso baseline with fixed alpha from config."""
    params = _model_config(config, "lasso")
    return build_week07_model_pipeline(
        Lasso(
            alpha=params.get("alpha", 0.01),
            max_iter=params.get("max_iter", 20000),
            random_state=params.get("random_state", config.get("random_state", 42)),
        ),
        X_train=X_train,
        config=config,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )


def build_elastic_net_model(
    X_train,
    config: dict,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> Pipeline:
    """Elastic Net baseline with fixed alpha and l1_ratio from config."""
    params = _model_config(config, "elastic_net")
    return build_week07_model_pipeline(
        ElasticNet(
            alpha=params.get("alpha", 0.01),
            l1_ratio=params.get("l1_ratio", 0.5),
            max_iter=params.get("max_iter", 20000),
            random_state=params.get("random_state", config.get("random_state", 42)),
        ),
        X_train=X_train,
        config=config,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )


def build_bayesian_ridge_model(
    X_train,
    config: dict,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> Pipeline:
    """Bayesian Ridge baseline with sklearn defaults unless config overrides them."""
    params = _model_config(config, "bayesian_ridge")
    return build_week07_model_pipeline(
        BayesianRidge(
            max_iter=params.get("max_iter", 300),
            tol=params.get("tol", 0.001),
        ),
        X_train=X_train,
        config=config,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )


def build_week07_models(
    X_train,
    config: dict,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> dict[str, Pipeline]:
    """Build all enabled fixed-parameter Week 7 Linear Family models."""
    registry = {
        "OLS": build_ols_model,
        "Ridge": build_ridge_model,
        "Lasso": build_lasso_model,
        "Elastic Net": build_elastic_net_model,
        "Bayesian Ridge": build_bayesian_ridge_model,
    }
    enabled = config.get("enabled_models", list(registry))

    return {
        model_name: registry[model_name](
            X_train=X_train,
            config=config,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )
        for model_name in enabled
        if model_name in registry
    }
