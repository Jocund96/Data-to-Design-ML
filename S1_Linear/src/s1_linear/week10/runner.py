"""Run Week 10 feature attribution and held-out permutation importance."""

from pathlib import Path
import argparse
import os
import tempfile

cache_root = Path(tempfile.gettempdir()) / "s1_linear_plot_cache"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
os.environ.setdefault("MPLBACKEND", "Agg")

from s1_linear.config import load_config
from s1_linear.week10.agreement import compare_feature_rankings
from s1_linear.week10.data import (
    load_week10_inputs,
    make_readiness_audit,
    resolve_project_path,
)
from s1_linear.week10.feature_mapping import (
    make_feature_group_mapping,
    map_transformed_to_original,
    merge_transformed_and_group_mapping,
)
from s1_linear.week10.permutation import (
    run_feature_permutation_importance,
    run_group_permutation_importance,
)
from s1_linear.week10.plots import (
    plot_feature_permutation,
    plot_group_permutation,
    plot_group_shap,
    plot_rank_agreement,
    plot_top_original_shap,
)
from s1_linear.week10.shap_analysis import run_shap_attribution


project_root = Path(__file__).resolve().parents[3]


def _save_csv(frame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def main(config_path: str) -> None:
    """Run Week 10 attribution on the shared publication-held-out test set."""
    config = load_config(resolve_project_path(project_root, config_path))
    output_config = config["outputs"]
    directories = {
        key: resolve_project_path(project_root, output_config[key])
        for key in ("tables_dir", "figures_dir", "predictions_dir", "metrics_dir")
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    inputs = load_week10_inputs(config, project_root)
    week09 = inputs.week09_inputs
    fitted_pipeline = inputs.fitted_model
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    model = fitted_pipeline.named_steps["model"]

    transformed_features = list(preprocessor.get_feature_names_out())
    raw_features = week09.X_test.columns.tolist()
    feature_group_mapping = make_feature_group_mapping(
        raw_features=raw_features,
        configured_groups=config["feature_groups"],
    )
    transformed_mapping = map_transformed_to_original(
        transformed_features=transformed_features,
        original_features=raw_features,
    )
    transformed_group_mapping = merge_transformed_and_group_mapping(
        transformed_mapping,
        feature_group_mapping,
    )

    shap_result = run_shap_attribution(
        fitted_pipeline=fitted_pipeline,
        X_background=week09.X_train,
        X_evaluation=week09.X_test,
        transformed_mapping=transformed_group_mapping,
        feature_group_mapping=feature_group_mapping,
        prefer_shap_package=bool(config["shap"].get("prefer_shap_package", True)),
    )
    additive_max_abs_error = float(
        shap_result.prediction_audit["absolute_additivity_error"].max()
    )
    readiness = make_readiness_audit(
        inputs=inputs,
        transformed_feature_count=len(transformed_features),
        model_coefficient_count=len(model.coef_),
        shap_method=shap_result.method,
        shap_available=shap_result.shap_available,
        additive_max_abs_error=additive_max_abs_error,
        additive_tolerance=float(config["shap"]["additive_tolerance"]),
    )

    feature_perm, feature_perm_repeats = run_feature_permutation_importance(
        fitted_pipeline=fitted_pipeline,
        X_test=week09.X_test,
        y_test=week09.y_test,
        features=raw_features,
        n_repeats=int(config["permutation"]["n_repeats"]),
        random_state=int(config["permutation"]["random_state"]),
    )
    feature_perm = feature_perm.merge(
        feature_group_mapping,
        on="original_feature",
        how="left",
        validate="one_to_one",
    )
    group_perm, group_perm_repeats = run_group_permutation_importance(
        fitted_pipeline=fitted_pipeline,
        X_test=week09.X_test,
        y_test=week09.y_test,
        feature_group_mapping=feature_group_mapping,
        n_repeats=int(config["permutation"]["n_repeats"]),
        random_state=int(config["permutation"]["random_state"]) + 10_000,
    )
    agreement = compare_feature_rankings(
        shap_feature_importance=shap_result.original_importance,
        permutation_feature_importance=feature_perm,
        shap_group_importance=shap_result.group_importance,
        permutation_group_importance=group_perm,
        top_k_features=int(config["agreement"]["top_k_features"]),
        top_k_groups=int(config["agreement"]["top_k_groups"]),
    )

    tables_dir = directories["tables_dir"]
    predictions_dir = directories["predictions_dir"]
    metrics_dir = directories["metrics_dir"]
    figures_dir = directories["figures_dir"]
    _save_csv(readiness, tables_dir / output_config["readiness_audit_name"])
    _save_csv(
        transformed_group_mapping,
        tables_dir / output_config["transformed_mapping_name"],
    )
    _save_csv(
        feature_group_mapping,
        tables_dir / output_config["feature_group_mapping_name"],
    )
    _save_csv(
        shap_result.prediction_audit,
        tables_dir / output_config["shap_prediction_audit_name"],
    )
    _save_csv(
        shap_result.transformed_importance,
        tables_dir / output_config["shap_transformed_importance_name"],
    )
    _save_csv(
        shap_result.original_importance,
        tables_dir / output_config["shap_original_importance_name"],
    )
    _save_csv(
        shap_result.group_importance,
        tables_dir / output_config["shap_group_importance_name"],
    )
    _save_csv(
        shap_result.original_values,
        predictions_dir / output_config["shap_original_values_name"],
    )
    _save_csv(
        shap_result.group_values,
        predictions_dir / output_config["shap_group_values_name"],
    )
    _save_csv(feature_perm, metrics_dir / output_config["feature_permutation_name"])
    _save_csv(
        feature_perm_repeats,
        metrics_dir / output_config["feature_permutation_repeats_name"],
    )
    _save_csv(group_perm, metrics_dir / output_config["group_permutation_name"])
    _save_csv(
        group_perm_repeats,
        metrics_dir / output_config["group_permutation_repeats_name"],
    )
    _save_csv(agreement, tables_dir / output_config["shap_permutation_agreement_name"])

    plot_top_original_shap(
        shap_result.original_importance,
        figures_dir / output_config["shap_original_figure"],
    )
    plot_group_shap(
        shap_result.group_importance,
        figures_dir / output_config["shap_group_figure"],
    )
    plot_feature_permutation(
        feature_perm,
        figures_dir / output_config["feature_permutation_figure"],
    )
    plot_group_permutation(
        group_perm,
        figures_dir / output_config["group_permutation_figure"],
    )
    plot_rank_agreement(
        shap_result.original_importance,
        feature_perm,
        figures_dir / output_config["rank_agreement_figure"],
    )

    print("Week 10 feature attribution complete.")
    print("\nReadiness audit:")
    print(readiness[["check", "value", "status"]].to_string(index=False))
    print("\nTop SHAP original features:")
    print(
        shap_result.original_importance[
            ["original_feature", "feature_group", "mean_abs_shap", "shap_rank"]
        ]
        .head(10)
        .round(3)
        .to_string(index=False)
    )
    print("\nSHAP group importance:")
    print(
        shap_result.group_importance[
            [
                "feature_group",
                "mean_abs_shap",
                "sum_mean_abs_original_feature_shap",
                "shap_rank",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )
    print("\nTop held-out feature permutation importance:")
    print(
        feature_perm[
            ["original_feature", "feature_group", "mean_rmse_delta", "permutation_rank"]
        ]
        .head(10)
        .round(3)
        .to_string(index=False)
    )
    print("\nHeld-out group permutation importance:")
    print(
        group_perm[
            ["feature_group", "mean_rmse_delta", "permutation_rank"]
        ]
        .round(3)
        .to_string(index=False)
    )
    print("\nSHAP/permutation agreement:")
    print(agreement.round(3).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Week 10 feature attribution experiments."
    )
    parser.add_argument(
        "--config",
        default="configs/week10_feature_attribution.yaml",
        help="Config path relative to S1_Linear.",
    )
    args = parser.parse_args()
    main(args.config)

