"""Shared-test and LOPO uncertainty experiments for Week 9."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from s1_linear.week09.calibration import (
    classify_publication_confidence,
    conformal_quantile_audit,
    conformalize_interval_bounds,
    finite_sample_conformal_quantile,
    interval_metrics,
    make_interval_prediction_frame,
    publication_interval_metrics,
    summarize_micro_macro,
    symmetric_interval,
)
from s1_linear.week09.data import Week09Inputs
from s1_linear.week09.methods import (
    bootstrap_percentile_interval,
    fit_frozen_pipeline,
    normal_interval,
    predict_bayesian_distribution,
    residual_bootstrap_prediction_distributions,
)
from s1_linear.week09.splits import (
    audit_role_manifest,
    eligible_lopo_publications,
    make_lopo_role_manifest,
    split_modeling_data_by_role,
)


@dataclass
class IntervalExperimentResult:
    """Tables and fitted models produced by one calibrated evaluation."""

    predictions: pd.DataFrame
    metrics: pd.DataFrame
    publication_metrics: pd.DataFrame
    coverage_curve: pd.DataFrame
    calibration_quantiles: pd.DataFrame
    fitted_models: dict[str, object]


def _append_lineage(
    predictions: pd.DataFrame,
    lineage: pd.DataFrame,
) -> pd.DataFrame:
    if len(predictions) != len(lineage):
        raise ValueError("Prediction and lineage rows must align.")
    duplicate_columns = set(predictions.columns) & set(lineage.columns)
    if duplicate_columns:
        lineage = lineage.drop(columns=sorted(duplicate_columns))
    return pd.concat(
        [predictions.reset_index(drop=True), lineage.reset_index(drop=True)],
        axis=1,
    )


def _quantile_row(
    method: str,
    model_name: str,
    evaluation_scheme: str,
    evaluation_fold: str,
    score_type: str,
    audit: dict,
) -> dict:
    return {
        "method": method,
        "model": model_name,
        "evaluation_scheme": evaluation_scheme,
        "evaluation_fold": evaluation_fold,
        "calibration_score_type": score_type,
        **audit,
    }


def run_interval_methods(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    train_lineage: pd.DataFrame,
    X_calibration: pd.DataFrame,
    y_calibration: pd.Series,
    calibration_lineage: pd.DataFrame,
    X_evaluation: pd.DataFrame,
    y_evaluation: pd.Series,
    evaluation_lineage: pd.DataFrame,
    frozen_config: dict,
    policy: str,
    calibration_config: dict,
    bootstrap_config: dict,
    evaluation_scheme: str,
    evaluation_fold: str,
    split_name: str,
    random_state_offset: int = 0,
) -> IntervalExperimentResult:
    """Fit and evaluate all configured interval methods without target leakage."""
    alpha = float(calibration_config["alpha"])
    primary_coverage = 1 - alpha
    coverage_levels = sorted(
        set(float(value) for value in calibration_config["coverage_levels"])
        | {primary_coverage}
    )
    epsilon = float(calibration_config.get("epsilon", 1e-8))
    enabled = calibration_config["methods"]
    prediction_frames = []
    metric_rows = []
    curve_rows = []
    quantile_rows = []
    fitted_models = {}

    common_extra = {
        "policy": policy,
        "evaluation_fold": evaluation_fold,
        "training_rows": len(X_train),
        "training_publications": int(train_lineage["publication_group"].nunique()),
        "calibration_rows": len(X_calibration),
        "calibration_publications": int(
            calibration_lineage["publication_group"].nunique()
        ),
        "evaluation_rows": len(X_evaluation),
        "evaluation_publications": int(
            evaluation_lineage["publication_group"].nunique()
        ),
    }

    def record_interval(
        method: str,
        model_name: str,
        nominal_coverage: float,
        prediction,
        lower,
        upper,
        uncertainty_score,
    ) -> None:
        frame = make_interval_prediction_frame(
            y_true=y_evaluation,
            prediction=prediction,
            lower=lower,
            upper=upper,
            method=method,
            model_name=model_name,
            evaluation_scheme=evaluation_scheme,
            split=split_name,
            nominal_coverage=nominal_coverage,
            uncertainty_score=uncertainty_score,
            epsilon=epsilon,
            **common_extra,
        )
        metrics = interval_metrics(frame)
        metrics["evaluation_fold"] = evaluation_fold
        curve_rows.append(metrics)
        if np.isclose(nominal_coverage, primary_coverage):
            frame = _append_lineage(frame, evaluation_lineage)
            prediction_frames.append(frame)
            metric_rows.append(metrics)

    elastic_required = any(
        enabled.get(name, False)
        for name in (
            "elastic_net_split_conformal",
            "elastic_net_residual_bootstrap",
            "elastic_net_bootstrap_conformalized",
        )
    )
    elastic_model = None
    elastic_calibration_prediction = None
    elastic_evaluation_prediction = None
    if elastic_required:
        elastic_model, _ = fit_frozen_pipeline(
            "Elastic Net",
            frozen_config,
            X_train,
            y_train,
        )
        fitted_models["Elastic Net"] = elastic_model
        elastic_calibration_prediction = elastic_model.predict(X_calibration)
        elastic_evaluation_prediction = elastic_model.predict(X_evaluation)

    if enabled.get("elastic_net_split_conformal", False):
        calibration_scores = np.abs(
            np.asarray(y_calibration, dtype=float)
            - np.asarray(elastic_calibration_prediction, dtype=float)
        )
        for nominal_coverage in coverage_levels:
            method_alpha = 1 - nominal_coverage
            q_hat = finite_sample_conformal_quantile(
                calibration_scores,
                method_alpha,
            )
            lower, upper = symmetric_interval(elastic_evaluation_prediction, q_hat)
            record_interval(
                method="elastic_net_split_conformal",
                model_name="Elastic Net",
                nominal_coverage=nominal_coverage,
                prediction=elastic_evaluation_prediction,
                lower=lower,
                upper=upper,
                uncertainty_score=np.full(len(X_evaluation), q_hat),
            )
            quantile_rows.append(
                _quantile_row(
                    method="elastic_net_split_conformal",
                    model_name="Elastic Net",
                    evaluation_scheme=evaluation_scheme,
                    evaluation_fold=evaluation_fold,
                    score_type="absolute_validation_residual",
                    audit=conformal_quantile_audit(
                        calibration_scores,
                        method_alpha,
                    ),
                )
            )

    bayesian_required = enabled.get("bayesian_ridge_native", False) or enabled.get(
        "bayesian_ridge_conformalized",
        False,
    )
    if bayesian_required:
        bayesian_model, _ = fit_frozen_pipeline(
            "Bayesian Ridge",
            frozen_config,
            X_train,
            y_train,
        )
        fitted_models["Bayesian Ridge"] = bayesian_model
        bayesian_calibration_mean, bayesian_calibration_std = (
            predict_bayesian_distribution(bayesian_model, X_calibration)
        )
        bayesian_evaluation_mean, bayesian_evaluation_std = (
            predict_bayesian_distribution(bayesian_model, X_evaluation)
        )

        if enabled.get("bayesian_ridge_native", False):
            for nominal_coverage in coverage_levels:
                lower, upper, z_value = normal_interval(
                    bayesian_evaluation_mean,
                    bayesian_evaluation_std,
                    nominal_coverage,
                )
                record_interval(
                    method="bayesian_ridge_native",
                    model_name="Bayesian Ridge",
                    nominal_coverage=nominal_coverage,
                    prediction=bayesian_evaluation_mean,
                    lower=lower,
                    upper=upper,
                    uncertainty_score=bayesian_evaluation_std,
                )
                quantile_rows.append(
                    {
                        "method": "bayesian_ridge_native",
                        "model": "Bayesian Ridge",
                        "evaluation_scheme": evaluation_scheme,
                        "evaluation_fold": evaluation_fold,
                        "calibration_score_type": "gaussian_predictive_z",
                        "alpha": 1 - nominal_coverage,
                        "nominal_coverage": nominal_coverage,
                        "n_calibration_rows": len(X_calibration),
                        "finite_sample_rank": np.nan,
                        "quantile_level": np.nan,
                        "q_hat": z_value,
                    }
                )

        if enabled.get("bayesian_ridge_conformalized", False):
            normalized_scores = np.abs(
                np.asarray(y_calibration, dtype=float) - bayesian_calibration_mean
            ) / np.maximum(bayesian_calibration_std, epsilon)
            for nominal_coverage in coverage_levels:
                method_alpha = 1 - nominal_coverage
                q_hat = finite_sample_conformal_quantile(
                    normalized_scores,
                    method_alpha,
                )
                lower, upper = symmetric_interval(
                    bayesian_evaluation_mean,
                    q_hat * np.maximum(bayesian_evaluation_std, epsilon),
                )
                record_interval(
                    method="bayesian_ridge_conformalized",
                    model_name="Bayesian Ridge",
                    nominal_coverage=nominal_coverage,
                    prediction=bayesian_evaluation_mean,
                    lower=lower,
                    upper=upper,
                    uncertainty_score=bayesian_evaluation_std,
                )
                quantile_rows.append(
                    _quantile_row(
                        method="bayesian_ridge_conformalized",
                        model_name="Bayesian Ridge",
                        evaluation_scheme=evaluation_scheme,
                        evaluation_fold=evaluation_fold,
                        score_type="absolute_residual_over_predictive_std",
                        audit=conformal_quantile_audit(
                            normalized_scores,
                            method_alpha,
                        ),
                    )
                )

    bootstrap_required = enabled.get(
        "elastic_net_residual_bootstrap",
        False,
    ) or enabled.get("elastic_net_bootstrap_conformalized", False)
    if bootstrap_required:
        repetitions = int(bootstrap_config["repetitions"])
        distributions = residual_bootstrap_prediction_distributions(
            fitted_elastic_net_pipeline=elastic_model,
            X_train=X_train,
            y_train=y_train,
            prediction_sets={
                "calibration": X_calibration,
                "evaluation": X_evaluation,
            },
            repetitions=repetitions,
            random_state=int(bootstrap_config["random_state"]) + random_state_offset,
        )
        bootstrap_uncertainty = distributions["evaluation"].std(axis=0, ddof=0)

        for nominal_coverage in coverage_levels:
            calibration_lower, calibration_upper = bootstrap_percentile_interval(
                distributions["calibration"],
                nominal_coverage,
            )
            evaluation_lower, evaluation_upper = bootstrap_percentile_interval(
                distributions["evaluation"],
                nominal_coverage,
            )

            if enabled.get("elastic_net_residual_bootstrap", False):
                record_interval(
                    method="elastic_net_residual_bootstrap",
                    model_name="Elastic Net",
                    nominal_coverage=nominal_coverage,
                    prediction=elastic_evaluation_prediction,
                    lower=evaluation_lower,
                    upper=evaluation_upper,
                    uncertainty_score=bootstrap_uncertainty,
                )
                quantile_rows.append(
                    {
                        "method": "elastic_net_residual_bootstrap",
                        "model": "Elastic Net",
                        "evaluation_scheme": evaluation_scheme,
                        "evaluation_fold": evaluation_fold,
                        "calibration_score_type": "bootstrap_percentile_bounds",
                        "alpha": 1 - nominal_coverage,
                        "nominal_coverage": nominal_coverage,
                        "n_calibration_rows": len(X_calibration),
                        "finite_sample_rank": np.nan,
                        "quantile_level": np.nan,
                        "q_hat": np.nan,
                        "bootstrap_repetitions": repetitions,
                    }
                )

            if enabled.get("elastic_net_bootstrap_conformalized", False):
                adjusted_lower, adjusted_upper, q_hat = conformalize_interval_bounds(
                    y_calibration=y_calibration,
                    calibration_lower=calibration_lower,
                    calibration_upper=calibration_upper,
                    evaluation_lower=evaluation_lower,
                    evaluation_upper=evaluation_upper,
                    alpha=1 - nominal_coverage,
                )
                record_interval(
                    method="elastic_net_bootstrap_conformalized",
                    model_name="Elastic Net",
                    nominal_coverage=nominal_coverage,
                    prediction=elastic_evaluation_prediction,
                    lower=adjusted_lower,
                    upper=adjusted_upper,
                    uncertainty_score=bootstrap_uncertainty,
                )
                nonconformity = np.maximum.reduce(
                    [
                        calibration_lower - np.asarray(y_calibration, dtype=float),
                        np.asarray(y_calibration, dtype=float) - calibration_upper,
                        np.zeros(len(y_calibration)),
                    ]
                )
                audit = conformal_quantile_audit(
                    nonconformity,
                    1 - nominal_coverage,
                )
                audit["q_hat"] = q_hat
                quantile_rows.append(
                    _quantile_row(
                        method="elastic_net_bootstrap_conformalized",
                        model_name="Elastic Net",
                        evaluation_scheme=evaluation_scheme,
                        evaluation_fold=evaluation_fold,
                        score_type="outside_bootstrap_interval_distance",
                        audit=audit,
                    )
                )

    if not prediction_frames:
        raise ValueError("No Week 9 interval methods are enabled.")
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    publication_metrics = publication_interval_metrics(predictions)
    coverage_curve = pd.DataFrame(curve_rows)
    calibration_quantiles = pd.DataFrame(quantile_rows)
    return IntervalExperimentResult(
        predictions=predictions,
        metrics=metrics,
        publication_metrics=publication_metrics,
        coverage_curve=coverage_curve,
        calibration_quantiles=calibration_quantiles,
        fitted_models=fitted_models,
    )


def run_shared_calibration_experiment(
    inputs: Week09Inputs,
    config: dict,
) -> IntervalExperimentResult:
    """Run Week 9 intervals on the frozen Week 8 shared publication split."""
    return run_interval_methods(
        X_train=inputs.X_train,
        y_train=inputs.y_train,
        train_lineage=inputs.train_lineage,
        X_calibration=inputs.X_calibration,
        y_calibration=inputs.y_calibration,
        calibration_lineage=inputs.calibration_lineage,
        X_evaluation=inputs.X_test,
        y_evaluation=inputs.y_test,
        evaluation_lineage=inputs.test_lineage,
        frozen_config=inputs.frozen_config,
        policy=inputs.policy,
        calibration_config=config["calibration"],
        bootstrap_config=config["bootstrap"],
        evaluation_scheme="shared_publication_held_out",
        evaluation_fold="shared_publication_test",
        split_name="test",
    )


def run_lopo_calibration_experiment(
    inputs: Week09Inputs,
    config: dict,
) -> dict[str, pd.DataFrame]:
    """Run calibrated intervals for every thresholded LOPO publication."""
    eligible = eligible_lopo_publications(
        inputs.modeling_lineage,
        inputs.minimum_lopo_rows,
    )
    manifests = []
    role_summaries = []
    leakage_audits = []
    predictions = []
    publication_metrics_frames = []
    quantile_frames = []
    coverage_curve_frames = []

    for fold_index, heldout_publication in enumerate(eligible["publication_group"]):
        manifest = make_lopo_role_manifest(
            lineage_df=inputs.modeling_lineage,
            week08_manifest=inputs.split_manifest,
            heldout_publication=heldout_publication,
            calibration_manifest_split=config["lopo"]["calibration_manifest_split"],
        )
        role_summary, leakage_audit = audit_role_manifest(
            lineage_df=inputs.modeling_lineage,
            role_manifest=manifest,
            expected_heldout_publication=heldout_publication,
        )
        role_data = split_modeling_data_by_role(
            modeling_df=inputs.modeling_df,
            lineage_df=inputs.modeling_lineage,
            role_manifest=manifest,
            target_col=inputs.target_col,
        )
        X_train, y_train, train_lineage = role_data["train"]
        X_calibration, y_calibration, calibration_lineage = role_data["calibration"]
        X_heldout, y_heldout, heldout_lineage = role_data["heldout"]
        result = run_interval_methods(
            X_train=X_train,
            y_train=y_train,
            train_lineage=train_lineage,
            X_calibration=X_calibration,
            y_calibration=y_calibration,
            calibration_lineage=calibration_lineage,
            X_evaluation=X_heldout,
            y_evaluation=y_heldout,
            evaluation_lineage=heldout_lineage,
            frozen_config=inputs.frozen_config,
            policy=inputs.policy,
            calibration_config=config["calibration"],
            bootstrap_config=config["bootstrap"],
            evaluation_scheme="leave_one_publication_out",
            evaluation_fold=heldout_publication,
            split_name="heldout_publication",
            random_state_offset=(fold_index + 1) * 1000,
        )
        manifests.append(manifest)
        role_summaries.append(role_summary)
        leakage_audits.append(leakage_audit)
        predictions.append(result.predictions)
        publication_metrics_frames.append(result.publication_metrics)
        quantile_frames.append(result.calibration_quantiles)
        coverage_curve_frames.append(result.coverage_curve)

    all_predictions = pd.concat(predictions, ignore_index=True)
    publication_metrics_df = pd.concat(
        publication_metrics_frames,
        ignore_index=True,
    )
    confidence = classify_publication_confidence(publication_metrics_df)
    return {
        "eligible_publications": eligible,
        "role_manifest": pd.concat(manifests, ignore_index=True),
        "role_summary": pd.concat(role_summaries, ignore_index=True),
        "leakage_audit": pd.concat(leakage_audits, ignore_index=True),
        "predictions": all_predictions,
        "publication_metrics": publication_metrics_df,
        "micro_macro": summarize_micro_macro(all_predictions),
        "confidence_diagnostics": confidence,
        "calibration_quantiles": pd.concat(quantile_frames, ignore_index=True),
        "coverage_curve": pd.concat(coverage_curve_frames, ignore_index=True),
    }

