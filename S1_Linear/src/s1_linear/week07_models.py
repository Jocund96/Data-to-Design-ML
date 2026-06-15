"""Linear Family model builders for Week 7 UHPC experiments."""

from sklearn.linear_model import BayesianRidge, ElasticNet, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from s1_linear.week07_preprocessing import build_week07_preprocessor


MODEL_NAME_TO_CONFIG_KEY = {
    "OLS": "ols",
    "Elastic Net": "elastic_net",
    "Bayesian Ridge": "bayesian_ridge",
    "Polynomial Ridge": "polynomial_ridge",
}


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
            max_iter=params.get("max_iter", 50000),
            tol=params.get("tol", 0.0001),
            selection=params.get("selection", "cyclic"),
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
            alpha_1=params.get("alpha_1", 1e-6),
            alpha_2=params.get("alpha_2", 1e-6),
            lambda_1=params.get("lambda_1", 1e-6),
            lambda_2=params.get("lambda_2", 1e-6),
        ),
        X_train=X_train,
        config=config,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )


def build_polynomial_ridge_model(
    X_train,
    config: dict,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> Pipeline:
    """
    Build Polynomial Ridge with expansion after train-fitted preprocessing.

    Scaling after polynomial expansion keeps the Ridge penalty comparable
    across original, squared, and interaction terms.
    """
    params = _model_config(config, "polynomial_ridge")
    base_pipeline = build_week07_model_pipeline(
        Ridge(
            alpha=params.get("alpha", 10.0),
            solver=params.get("solver", "lsqr"),
            max_iter=params.get("max_iter", 5000),
            tol=params.get("tol", 0.0001),
        ),
        X_train=X_train,
        config=config,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    preprocessor = base_pipeline.named_steps["preprocessor"]

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "poly",
                PolynomialFeatures(
                    degree=params.get("degree", 2),
                    include_bias=False,
                    interaction_only=params.get("interaction_only", False),
                ),
            ),
            ("polynomial_scaler", StandardScaler()),
            ("model", base_pipeline.named_steps["model"]),
        ]
    )


def build_week07_models(
    X_train,
    config: dict,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> dict[str, Pipeline]:
    """Build all enabled Week 7 Linear Family models."""
    registry = {
        "OLS": build_ols_model,
        "Elastic Net": build_elastic_net_model,
        "Bayesian Ridge": build_bayesian_ridge_model,
        "Polynomial Ridge": build_polynomial_ridge_model,
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
