import numpy as np
import pandas as pd

from s1_linear.week09.calibration import (
    classify_publication_confidence,
    conformalize_interval_bounds,
    finite_sample_conformal_quantile,
    interval_metrics,
    make_interval_prediction_frame,
)
from s1_linear.week09.methods import (
    fit_frozen_pipeline,
    residual_bootstrap_prediction_distributions,
)
from s1_linear.week09.splits import (
    audit_role_manifest,
    make_lopo_role_manifest,
)


def _elastic_config():
    return {
        "random_state": 42,
        "enabled_models": ["Elastic Net"],
        "hyperparameter_tuning": {"enabled": False},
        "preprocessing": {
            "numeric_add_indicator": True,
            "categorical_missing_value": "missing_reported_gap",
            "categorical_min_frequency": 1,
            "categorical_max_categories": 10,
        },
        "models": {
            "elastic_net": {
                "alpha": 0.01,
                "l1_ratio": 0.7,
                "max_iter": 5000,
                "tol": 0.0001,
                "random_state": 42,
            }
        },
    }


def test_finite_sample_conformal_quantile_uses_higher_order_statistic():
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    q_hat = finite_sample_conformal_quantile(scores, alpha=0.20)

    assert q_hat == 5.0


def test_interval_metrics_match_hand_calculation():
    frame = make_interval_prediction_frame(
        y_true=[0.0, 5.0, 10.0],
        prediction=[1.0, 5.0, 8.0],
        lower=[0.0, 4.0, 7.0],
        upper=[2.0, 6.0, 9.0],
        method="test_method",
        model_name="test_model",
        evaluation_scheme="test_scheme",
        split="test",
        nominal_coverage=0.90,
    )
    metrics = interval_metrics(frame)

    assert np.isclose(metrics["EmpiricalCoverage"], 2 / 3)
    assert np.isclose(metrics["MeanIntervalWidth"], 2.0)
    assert np.isclose(metrics["MeanWinklerScore"], (2 + 2 + 22) / 3)
    assert np.isclose(metrics["BelowIntervalRate"], 0.0)
    assert np.isclose(metrics["AboveIntervalRate"], 1 / 3)
    assert np.isnan(metrics["UncertaintyErrorSpearman"])


def test_conformalized_base_interval_expands_by_validation_score():
    lower, upper, q_hat = conformalize_interval_bounds(
        y_calibration=[0.0, 3.0, 10.0],
        calibration_lower=[0.0, 2.0, 6.0],
        calibration_upper=[1.0, 4.0, 8.0],
        evaluation_lower=[5.0, 7.0],
        evaluation_upper=[6.0, 8.0],
        alpha=0.25,
    )

    assert q_hat == 2.0
    np.testing.assert_allclose(lower, [3.0, 5.0])
    np.testing.assert_allclose(upper, [8.0, 10.0])


def test_lopo_manifest_has_disjoint_publication_roles():
    lineage = pd.DataFrame(
        {
            "publication_group": [
                "paper_a",
                "paper_a",
                "paper_b",
                "paper_b",
                "paper_c",
                "paper_d",
            ]
        }
    )
    week08_manifest = pd.DataFrame(
        {
            "publication_group": ["paper_a", "paper_b", "paper_c", "paper_d"],
            "split": ["train", "validation", "test", "train"],
        }
    )
    manifest = make_lopo_role_manifest(
        lineage_df=lineage,
        week08_manifest=week08_manifest,
        heldout_publication="paper_a",
    )
    summary, audit = audit_role_manifest(
        lineage_df=lineage,
        role_manifest=manifest,
        expected_heldout_publication="paper_a",
    )

    assert set(audit["status"]) == {"pass"}
    assert set(summary["role"]) == {"train", "calibration", "heldout"}
    assert manifest.loc[
        manifest["publication_group"].eq("paper_a"),
        "week09_role",
    ].item() == "heldout"


def test_constant_width_diagnostics_ignore_floating_point_dust():
    metrics = pd.DataFrame(
        {
            "method": ["constant", "constant"],
            "publication_group": ["paper_a", "paper_b"],
            "n_rows": [5, 20],
            "nominal_coverage": [0.90, 0.90],
            "EmpiricalCoverage": [0.50, 1.00],
            "MeanIntervalWidth": [100.0, 100.0 + 1e-10],
        }
    )

    result = classify_publication_confidence(metrics)

    assert "wide_and_wrong" not in set(result["ConfidenceDiagnostic"])
    assert "wide_but_safe" not in set(result["ConfidenceDiagnostic"])
    assert set(result["CoverageEvidence"]) == {
        "limited_n_lt_10",
        "more_stable_n_ge_10",
    }


def test_residual_bootstrap_is_deterministic_for_fixed_seed():
    X = pd.DataFrame(
        {
            "cement": [600.0, 650.0, 700.0, 750.0, 800.0, 850.0, 900.0, 950.0],
            "curing_method": ["standard", "water"] * 4,
        }
    )
    y = pd.Series([100.0, 108.0, 116.0, 121.0, 134.0, 141.0, 153.0, 160.0])
    model, _ = fit_frozen_pipeline("Elastic Net", _elastic_config(), X, y)

    first = residual_bootstrap_prediction_distributions(
        fitted_elastic_net_pipeline=model,
        X_train=X,
        y_train=y,
        prediction_sets={"evaluation": X.iloc[:3]},
        repetitions=6,
        random_state=7,
    )
    second = residual_bootstrap_prediction_distributions(
        fitted_elastic_net_pipeline=model,
        X_train=X,
        y_train=y,
        prediction_sets={"evaluation": X.iloc[:3]},
        repetitions=6,
        random_state=7,
    )

    np.testing.assert_allclose(first["evaluation"], second["evaluation"])
