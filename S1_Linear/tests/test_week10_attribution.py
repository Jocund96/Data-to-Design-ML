import numpy as np
import pandas as pd

from s1_linear.week10.feature_mapping import map_transformed_to_original
from s1_linear.week10.permutation import run_feature_permutation_importance
from s1_linear.week10.shap_analysis import compute_linear_shap_values


class _Preprocessor:
    def transform(self, X):
        return X[["x1", "x2"]].to_numpy(dtype=float)


class _LinearModel:
    coef_ = np.array([2.0, -1.0])
    intercept_ = 3.0

    def predict(self, X):
        return self.intercept_ + np.asarray(X, dtype=float) @ self.coef_


class _Pipeline:
    named_steps = {"preprocessor": _Preprocessor(), "model": _LinearModel()}

    def predict(self, X):
        return self.named_steps["model"].predict(
            self.named_steps["preprocessor"].transform(X)
        )


def test_transformed_feature_mapping_handles_one_hot_names_with_underscores():
    mapping = map_transformed_to_original(
        transformed_features=[
            "num__cement",
            "target__cement_type",
            "ohe__fly_ash_type_class F",
            "ohe__curing_method_Water Curing",
        ],
        original_features=[
            "cement",
            "cement_type",
            "fly_ash_type",
            "curing_method",
        ],
    )

    assert mapping["original_feature"].tolist() == [
        "cement",
        "cement_type",
        "fly_ash_type",
        "curing_method",
    ]
    assert set(mapping["mapping_status"]) == {"mapped"}


def test_independent_linear_shap_formula_is_additive():
    X_background = pd.DataFrame({"x1": [0.0, 2.0], "x2": [1.0, 3.0]})
    X_evaluation = pd.DataFrame({"x1": [4.0, 5.0], "x2": [2.0, 4.0]})

    values, base_values, method, shap_available = compute_linear_shap_values(
        fitted_pipeline=_Pipeline(),
        X_background=X_background,
        X_evaluation=X_evaluation,
        transformed_features=["x1", "x2"],
        prefer_shap_package=False,
    )

    predictions = _Pipeline().predict(X_evaluation)
    reconstructed = base_values + values.sum(axis=1)
    assert method == "independent_linear_formula"
    assert shap_available is False
    np.testing.assert_allclose(reconstructed, predictions)


def test_feature_permutation_does_not_refit_and_returns_repeats():
    X = pd.DataFrame(
        {
            "x1": [0.0, 1.0, 2.0, 3.0, 4.0],
            "x2": [2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    y = pd.Series(_Pipeline().predict(X))

    summary, repeats = run_feature_permutation_importance(
        fitted_pipeline=_Pipeline(),
        X_test=X,
        y_test=y,
        features=["x1", "x2"],
        n_repeats=4,
        random_state=42,
    )

    assert len(repeats) == 8
    assert set(summary["original_feature"]) == {"x1", "x2"}
    assert (summary["baseline_rmse"] == 0).all()
