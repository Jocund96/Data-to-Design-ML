"""Load Week 10 inputs from the frozen Week 9 workflow."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from s1_linear.config import load_config
from s1_linear.week09.data import Week09Inputs, load_week09_inputs
from s1_linear.week09.methods import fit_frozen_pipeline


@dataclass
class Week10Inputs:
    """Validated Week 10 model, data, and lineage objects."""

    week09_inputs: Week09Inputs
    week09_readiness_audit: pd.DataFrame
    fitted_model: object
    fitted_config: dict
    model_name: str
    policy: str


def resolve_project_path(project_root: Path, path: str | Path) -> Path:
    """Resolve a config path relative to S1_Linear."""
    path = Path(path)
    return path if path.is_absolute() else (project_root / path).resolve()


def load_week10_inputs(config: dict, project_root: Path) -> Week10Inputs:
    """Load Week 9 artifacts and fit the frozen train-only model for attribution."""
    week09_config_path = resolve_project_path(
        project_root,
        config["data"]["week09_config_path"],
    )
    week09_config = load_config(week09_config_path)
    week09_inputs, week09_readiness = load_week09_inputs(
        week09_config,
        project_root,
    )
    model_name = config["model"]["model_name"]
    fitted_model, fitted_config = fit_frozen_pipeline(
        model_name=model_name,
        frozen_config=week09_inputs.frozen_config,
        X_train=week09_inputs.X_train,
        y_train=week09_inputs.y_train,
    )
    return Week10Inputs(
        week09_inputs=week09_inputs,
        week09_readiness_audit=week09_readiness,
        fitted_model=fitted_model,
        fitted_config=fitted_config,
        model_name=model_name,
        policy=config["data"]["policy"],
    )


def make_readiness_audit(
    inputs: Week10Inputs,
    transformed_feature_count: int,
    model_coefficient_count: int,
    shap_method: str,
    shap_available: bool,
    additive_max_abs_error: float,
    additive_tolerance: float,
) -> pd.DataFrame:
    """Create a Week 10 readiness audit table."""
    week09_failures = int(inputs.week09_readiness_audit["status"].eq("fail").sum())
    X_test = inputs.week09_inputs.X_test
    lineage = inputs.week09_inputs.test_lineage
    target = inputs.week09_inputs.target_col
    metadata_columns = {
        "semantic_row_id",
        "modeling_row_id",
        "mix_id",
        "publication_country",
        "publication_source",
        "publication_year",
        "publication_reference_id",
        "publication_group",
    }
    metadata_in_predictors = sorted(metadata_columns & set(X_test.columns))
    rows = [
        {
            "check": "week09_readiness_audit_passed",
            "value": week09_failures,
            "status": "pass" if week09_failures == 0 else "fail",
            "detail": "Week 10 reuses the frozen Week 9 input audit.",
        },
        {
            "check": "model_name",
            "value": inputs.model_name,
            "status": "pass" if inputs.model_name == "Elastic Net" else "warning",
            "detail": "Week 10 is configured to explain the Week 9 Elastic Net.",
        },
        {
            "check": "test_row_alignment",
            "value": f"{len(X_test)}/{len(inputs.week09_inputs.y_test)}/{len(lineage)}",
            "status": (
                "pass"
                if len(X_test) == len(inputs.week09_inputs.y_test) == len(lineage)
                else "fail"
            ),
            "detail": "X_test, y_test, and test lineage rows must align.",
        },
        {
            "check": "target_not_in_predictors",
            "value": target,
            "status": "pass" if target not in X_test.columns else "fail",
            "detail": "Target values are only used for evaluation.",
        },
        {
            "check": "lineage_columns_not_in_predictors",
            "value": len(metadata_in_predictors),
            "status": "pass" if not metadata_in_predictors else "fail",
            "detail": ", ".join(metadata_in_predictors),
        },
        {
            "check": "transformed_feature_count_matches_coefficients",
            "value": f"{transformed_feature_count}/{model_coefficient_count}",
            "status": (
                "pass"
                if transformed_feature_count == model_coefficient_count
                else "fail"
            ),
            "detail": "Each transformed feature needs exactly one linear coefficient.",
        },
        {
            "check": "shap_package_available",
            "value": shap_available,
            "status": "pass" if shap_available else "warning",
            "detail": (
                "SHAP is installed and shap.LinearExplainer was used."
                if shap_available
                else (
                    "SHAP is not installed or could not run; using the exact "
                    "independent linear SHAP formula."
                )
            ),
        },
        {
            "check": "shap_method_used",
            "value": shap_method,
            "status": "pass",
            "detail": "Method used for transformed-space SHAP values.",
        },
        {
            "check": "shap_additivity",
            "value": additive_max_abs_error,
            "status": (
                "pass"
                if additive_max_abs_error <= additive_tolerance
                else "fail"
            ),
            "detail": f"Tolerance: {additive_tolerance}",
        },
        {
            "check": "permutation_refits_model",
            "value": False,
            "status": "pass",
            "detail": "Week 10 permutation importance reuses the fitted model.",
        },
    ]
    return pd.DataFrame(rows)
