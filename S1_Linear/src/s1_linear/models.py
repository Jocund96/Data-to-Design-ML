from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, ElasticNet, BayesianRidge, Ridge


def build_ols_pipeline() -> Pipeline:
    """Ordinary Least Squares baseline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])


def build_elastic_net_pipeline(random_state: int = 42) -> Pipeline:
    """Elastic Net with L1/L2 regularization."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNet(max_iter=10000, random_state=random_state))
    ])


def build_bayesian_ridge_pipeline() -> Pipeline:
    """Bayesian Ridge regression baseline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", BayesianRidge())
    ])


def build_polynomial_ridge_pipeline() -> Pipeline:
    """Polynomial feature expansion followed by Ridge regression."""
    return Pipeline([
        ("poly", PolynomialFeatures(include_bias=False)),
        ("scaler", StandardScaler()),
        ("model", Ridge())
    ])
