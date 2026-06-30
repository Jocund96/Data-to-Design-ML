"""Run the complete Week 9 uncertainty and calibration workflow."""

from pathlib import Path
import argparse
import os
import tempfile

cache_root = Path(tempfile.gettempdir()) / "s1_linear_plot_cache"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
os.environ.setdefault("MPLBACKEND", "Agg")

import joblib
import pandas as pd
import yaml

from s1_linear.config import load_config
from s1_linear.week09.calibration import classify_publication_confidence
from s1_linear.week09.data import load_week09_inputs
from s1_linear.week09.experiments import (
    run_lopo_calibration_experiment,
    run_shared_calibration_experiment,
)
from s1_linear.week09.plots import (
    plot_coverage_curve,
    plot_coverage_vs_width,
    plot_interval_examples,
    plot_lopo_coverage,
    plot_lopo_shift_vs_uncertainty,
    plot_method_coverage_width,
    plot_publication_coverage,
    plot_uncertainty_vs_error,
)
from s1_linear.week09.splits import audit_role_manifest, make_shared_role_manifest


project_root = Path(__file__).resolve().parents[3]


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path relative to S1_Linear."""
    path = Path(path)
    return path if path.is_absolute() else (project_root / path).resolve()


def _save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _merge_week08_lopo_diagnostics(
    week09_metrics: pd.DataFrame,
    week08_metrics_path: Path,
) -> pd.DataFrame:
    """Join Week 8 point-error and shift diagnostics to Week 9 intervals."""
    week08 = pd.read_csv(week08_metrics_path)
    keep = [
        "publication_group",
        "MAE",
        "RMSE",
        "R2",
        "Bias",
        "MaximumAE",
        "numeric_out_of_training_range_rate",
        "unseen_category_rate",
        "worst_row_squared_error_share",
    ]
    week08 = week08[[column for column in keep if column in week08]].rename(
        columns={
            "MAE": "Week08PointMAE",
            "RMSE": "Week08PointRMSE",
            "R2": "Week08PointR2",
            "Bias": "Week08PointBias",
            "MaximumAE": "Week08MaximumAE",
        }
    )
    return week09_metrics.merge(
        week08,
        on="publication_group",
        how="left",
        validate="many_to_one",
    )


def main(config_path: str) -> None:
    """Run shared and thresholded-LOPO Week 9 calibration experiments."""
    config = load_config(resolve_project_path(config_path))
    output_config = config["outputs"]
    directories = {
        name: resolve_project_path(output_config[name])
        for name in (
            "data_dir",
            "tables_dir",
            "figures_dir",
            "models_dir",
            "metrics_dir",
            "predictions_dir",
        )
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    inputs, readiness_audit = load_week09_inputs(config, project_root)
    _save_csv(
        readiness_audit,
        directories["tables_dir"] / output_config["readiness_audit_name"],
    )

    shared_manifest = make_shared_role_manifest(inputs.split_manifest)
    shared_role_summary, shared_leakage_audit = audit_role_manifest(
        inputs.modeling_lineage,
        shared_manifest,
    )
    _save_csv(
        shared_manifest,
        directories["data_dir"] / output_config["shared_role_manifest_name"],
    )
    _save_csv(
        shared_role_summary,
        directories["tables_dir"] / output_config["shared_role_summary_name"],
    )
    _save_csv(
        shared_leakage_audit,
        directories["tables_dir"] / output_config["shared_leakage_audit_name"],
    )

    shared = run_shared_calibration_experiment(inputs, config)
    shared_confidence = classify_publication_confidence(shared.publication_metrics)
    shared_overconfident_rows = shared.predictions.loc[
        ~shared.predictions["Covered"]
    ].sort_values(["StandardizedError", "AbsoluteError"], ascending=False)

    _save_csv(
        shared.calibration_quantiles,
        directories["tables_dir"]
        / output_config["shared_calibration_quantiles_name"],
    )
    _save_csv(
        shared.metrics,
        directories["metrics_dir"] / output_config["shared_metrics_name"],
    )
    _save_csv(
        shared.publication_metrics,
        directories["tables_dir"]
        / output_config["shared_publication_metrics_name"],
    )
    _save_csv(
        shared.predictions,
        directories["predictions_dir"]
        / output_config["shared_predictions_name"],
    )
    _save_csv(
        shared.coverage_curve,
        directories["tables_dir"] / output_config["shared_coverage_curve_name"],
    )
    _save_csv(
        shared_confidence,
        directories["tables_dir"]
        / output_config["shared_confidence_diagnostics_name"],
    )
    _save_csv(
        shared_overconfident_rows,
        directories["tables_dir"]
        / output_config["shared_overconfident_rows_name"],
    )

    if "Elastic Net" in shared.fitted_models:
        joblib.dump(
            shared.fitted_models["Elastic Net"],
            directories["models_dir"] / output_config["elastic_net_model_name"],
        )
    if "Bayesian Ridge" in shared.fitted_models:
        joblib.dump(
            shared.fitted_models["Bayesian Ridge"],
            directories["models_dir"] / output_config["bayesian_ridge_model_name"],
        )
    frozen_week09 = {
        "week08_frozen_model_config": inputs.frozen_config,
        "week09_calibration": config["calibration"],
        "week09_bootstrap": config["bootstrap"],
        "minimum_lopo_rows_from_week08": inputs.minimum_lopo_rows,
    }
    with (
        directories["models_dir"] / output_config["frozen_config_name"]
    ).open("w", encoding="utf-8") as file:
        yaml.safe_dump(frozen_week09, file, sort_keys=False)

    lopo = None
    lopo_comparison = pd.DataFrame()
    if config["lopo"].get("enabled", True):
        lopo = run_lopo_calibration_experiment(inputs, config)
        _save_csv(
            lopo["role_manifest"],
            directories["data_dir"] / output_config["lopo_role_manifest_name"],
        )
        _save_csv(
            lopo["role_summary"],
            directories["tables_dir"] / output_config["lopo_role_summary_name"],
        )
        _save_csv(
            lopo["leakage_audit"],
            directories["tables_dir"] / output_config["lopo_leakage_audit_name"],
        )
        _save_csv(
            lopo["calibration_quantiles"],
            directories["tables_dir"]
            / output_config["lopo_calibration_quantiles_name"],
        )
        _save_csv(
            lopo["publication_metrics"],
            directories["metrics_dir"]
            / output_config["lopo_publication_metrics_name"],
        )
        _save_csv(
            lopo["micro_macro"],
            directories["tables_dir"] / output_config["lopo_micro_macro_name"],
        )
        _save_csv(
            lopo["predictions"],
            directories["predictions_dir"] / output_config["lopo_predictions_name"],
        )
        _save_csv(
            lopo["coverage_curve"],
            directories["tables_dir"] / output_config["lopo_coverage_curve_name"],
        )
        _save_csv(
            lopo["confidence_diagnostics"],
            directories["tables_dir"]
            / output_config["lopo_confidence_diagnostics_name"],
        )
        lopo_comparison = _merge_week08_lopo_diagnostics(
            lopo["publication_metrics"],
            resolve_project_path(config["data"]["week08_lopo_metrics_path"]),
        )
        _save_csv(
            lopo_comparison,
            directories["tables_dir"]
            / output_config["lopo_week08_comparison_name"],
        )

    plot_method_coverage_width(
        shared.metrics,
        directories["figures_dir"]
        / output_config["method_coverage_width_figure"],
    )
    plot_coverage_curve(
        shared.coverage_curve,
        directories["figures_dir"] / output_config["coverage_curve_figure"],
    )
    plot_publication_coverage(
        shared.publication_metrics,
        directories["figures_dir"] / output_config["publication_coverage_figure"],
    )
    plot_coverage_vs_width(
        shared.publication_metrics,
        directories["figures_dir"] / output_config["coverage_width_figure"],
    )
    plot_interval_examples(
        shared.predictions,
        shared.publication_metrics,
        directories["figures_dir"]
        / output_config["shared_interval_examples_figure"],
    )
    plot_uncertainty_vs_error(
        shared.predictions,
        directories["figures_dir"] / output_config["uncertainty_error_figure"],
    )
    if lopo is not None:
        plot_lopo_coverage(
            lopo["publication_metrics"],
            directories["figures_dir"] / output_config["lopo_coverage_figure"],
        )
        plot_lopo_shift_vs_uncertainty(
            lopo_comparison,
            directories["figures_dir"] / output_config["lopo_shift_figure"],
        )

    print("Week 9 uncertainty and calibration workflow complete.")
    print("\nInput readiness audit:")
    print(readiness_audit[["check", "value", "status"]].to_string(index=False))
    print("\nShared publication-held-out 90% interval metrics:")
    print(
        shared.metrics[
            [
                "method",
                "EmpiricalCoverage",
                "CoverageGap",
                "MeanIntervalWidth",
                "MeanWinklerScore",
                "MAE",
                "RMSE",
                "R2",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )
    if lopo is not None:
        print("\nThresholded LOPO micro/macro 90% interval summary:")
        print(
            lopo["micro_macro"][
                [
                    "method",
                    "aggregation",
                    "n_publications",
                    "n_rows",
                    "EmpiricalCoverage",
                    "MeanIntervalWidth",
                    "MeanWinklerScore",
                    "RMSE",
                    "R2",
                ]
            ]
            .round(3)
            .to_string(index=False)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Week 9 uncertainty and calibration experiments."
    )
    parser.add_argument(
        "--config",
        default="configs/week09_uncertainty_calibration.yaml",
        help="Config path relative to S1_Linear.",
    )
    args = parser.parse_args()
    main(args.config)

