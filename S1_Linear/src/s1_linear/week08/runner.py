"""Run the Week 8 publication-safe data linkage, audit, and shared split."""

from pathlib import Path
import argparse
import os
import tempfile

cache_root = Path(tempfile.gettempdir()) / "s1_linear_plot_cache"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd

from s1_linear.config import load_config
from s1_linear.week08.publication_data import (
    build_aligned_week08_data,
    make_linkage_audit_table,
    make_publication_audit,
)
from s1_linear.week08.experiments import (
    load_split,
    run_leave_one_publication_out,
    run_publication_model_selection,
    run_row_mixed_comparison,
)
from s1_linear.week08.splits import (
    make_size_balanced_publication_manifest,
    make_split_summary,
    save_publication_splits,
)
from s1_linear.week08.plots import (
    plot_lopo_bias,
    plot_lopo_rmse_ranking,
    plot_publication_group_sizes,
    plot_shift_diagnostics,
    plot_split_comparison,
    plot_unseen_predicted_actual,
)


project_root = Path(__file__).resolve().parents[3]


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a config path relative to S1_Linear."""
    path = Path(path)
    return path if path.is_absolute() else (project_root / path).resolve()


def main(config_path: str) -> None:
    """Build and validate the Week 8 publication-held-out data foundation."""
    config = load_config(resolve_project_path(config_path))
    data_config = config["data"]
    audit_config = config["publication_audit"]
    split_config = config["split"]
    modeling_config = config["modeling"]
    output_config = config["outputs"]

    paths = {
        "corrected_semantic": resolve_project_path(
            data_config["corrected_semantic_path"]
        ),
        "corrected_lineage": resolve_project_path(
            data_config["corrected_lineage_path"]
        ),
        "week07_linear_ready": resolve_project_path(
            data_config["week07_linear_ready_path"]
        ),
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Required {name} file not found: {path}")

    data_dir = resolve_project_path(output_config["data_dir"])
    split_dir = resolve_project_path(output_config["split_dir"])
    tables_dir = resolve_project_path(output_config["tables_dir"])
    models_dir = resolve_project_path(output_config["models_dir"])
    metrics_dir = resolve_project_path(output_config["metrics_dir"])
    predictions_dir = resolve_project_path(output_config["predictions_dir"])
    figures_dir = resolve_project_path(output_config["figures_dir"])
    for directory in (
        data_dir,
        split_dir,
        tables_dir,
        models_dir,
        metrics_dir,
        predictions_dir,
        figures_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    semantic_df = pd.read_csv(paths["corrected_semantic"])
    lineage_df = pd.read_csv(paths["corrected_lineage"])
    week07_linear_ready_df = pd.read_csv(paths["week07_linear_ready"])
    modeling_df, modeling_lineage, linkage_audit = build_aligned_week08_data(
        semantic_df=semantic_df,
        lineage_df=lineage_df,
        target_col=data_config["target"],
        drop_predictor_columns=data_config["drop_predictor_columns"],
        week07_linear_ready_df=week07_linear_ready_df,
    )

    modeling_df.to_csv(data_dir / output_config["modeling_data_name"], index=False)
    modeling_lineage.to_csv(
        data_dir / output_config["modeling_lineage_name"],
        index=False,
    )
    make_linkage_audit_table(linkage_audit, paths).to_csv(
        tables_dir / output_config["linkage_audit_name"],
        index=False,
    )

    publication_audit, publication_summary = make_publication_audit(
        modeling_df=modeling_df,
        lineage_df=modeling_lineage,
        target_col=data_config["target"],
        important_numeric_features=audit_config["important_numeric_features"],
        minimum_rows=audit_config["minimum_rows_for_reliable_metrics"],
    )
    publication_audit.to_csv(
        tables_dir / output_config["publication_audit_name"],
        index=False,
    )
    publication_summary.to_csv(
        tables_dir / output_config["publication_summary_name"],
        index=False,
    )

    manifest = make_size_balanced_publication_manifest(
        lineage_df=modeling_lineage,
        train_size=split_config["train_size"],
        validation_size=split_config["validation_size"],
        test_size=split_config["test_size"],
        random_state=split_config["random_state"],
        search_restarts=split_config["search_restarts"],
    )
    split_summary, leakage_audit = make_split_summary(
        modeling_df=modeling_df,
        lineage_df=modeling_lineage,
        manifest=manifest,
        target_col=data_config["target"],
    )
    manifest.to_csv(
        tables_dir / output_config["split_manifest_name"],
        index=False,
    )
    split_summary.to_csv(
        tables_dir / output_config["split_summary_name"],
        index=False,
    )
    leakage_audit.to_csv(
        tables_dir / output_config["split_leakage_audit_name"],
        index=False,
    )
    save_publication_splits(
        modeling_df=modeling_df,
        lineage_df=modeling_lineage,
        manifest=manifest,
        target_col=data_config["target"],
        split_dir=split_dir,
    )

    X_train, y_train, train_lineage = load_split(
        split_dir,
        "train",
        data_config["target"],
    )
    X_validation, y_validation, validation_lineage = load_split(
        split_dir,
        "validation",
        data_config["target"],
    )
    X_test, y_test, test_lineage = load_split(
        split_dir,
        "test",
        data_config["target"],
    )
    tuning_config = load_config(
        resolve_project_path(modeling_config["tuning_config_path"])
    )
    model_results = run_publication_model_selection(
        X_train=X_train,
        y_train=y_train,
        train_lineage=train_lineage,
        X_validation=X_validation,
        y_validation=y_validation,
        validation_lineage=validation_lineage,
        X_test=X_test,
        y_test=y_test,
        test_lineage=test_lineage,
        tuning_config=tuning_config,
        policy=data_config["policy"],
        models_dir=models_dir,
    )
    model_results["tuning_summary"].to_csv(
        tables_dir / output_config["tuning_summary_name"],
        index=False,
    )
    model_results["cv_results"].to_csv(
        tables_dir / output_config["tuning_cv_results_name"],
        index=False,
    )
    model_results["validation_metrics"].to_csv(
        metrics_dir / output_config["validation_metrics_name"],
        index=False,
    )
    model_results["validation_predictions"].to_csv(
        predictions_dir / output_config["validation_predictions_name"],
        index=False,
    )
    model_results["selected_summary"].to_csv(
        tables_dir / output_config["selected_model_name"],
        index=False,
    )
    model_results["final_test_metrics"].to_csv(
        metrics_dir / output_config["final_test_metrics_name"],
        index=False,
    )
    model_results["final_test_predictions"].to_csv(
        predictions_dir / output_config["final_test_predictions_name"],
        index=False,
    )

    split_comparison = run_row_mixed_comparison(
        selected_model_name=model_results["selected_model_name"],
        selected_config=model_results["selected_config"],
        row_mixed_split_dir=resolve_project_path(
            modeling_config["row_mixed_split_dir"]
        ),
        publication_test_metrics=model_results["final_test_metrics"],
        target_col=data_config["target"],
        policy=data_config["policy"],
    )
    split_comparison.to_csv(
        tables_dir / output_config["split_comparison_name"],
        index=False,
    )

    lopo_metrics = pd.DataFrame()
    lopo_summary = pd.DataFrame()
    if modeling_config.get("run_leave_one_publication_out", True):
        lopo_metrics, lopo_predictions, lopo_summary = run_leave_one_publication_out(
            modeling_df=modeling_df,
            lineage_df=modeling_lineage,
            target_col=data_config["target"],
            selected_model_name=model_results["selected_model_name"],
            selected_config=model_results["selected_config"],
            policy=data_config["policy"],
            minimum_rows=audit_config["minimum_rows_for_reliable_metrics"],
        )
        lopo_metrics.to_csv(
            metrics_dir / output_config["lopo_metrics_name"],
            index=False,
        )
        lopo_predictions.to_csv(
            predictions_dir / output_config["lopo_predictions_name"],
            index=False,
        )
        lopo_summary.to_csv(
            tables_dir / output_config["lopo_summary_name"],
            index=False,
        )
        lopo_metrics.head(20).to_csv(
            tables_dir / output_config["worst_publications_name"],
            index=False,
        )
        lopo_predictions.nlargest(50, "AbsoluteError").to_csv(
            tables_dir / output_config["worst_rows_name"],
            index=False,
        )

    plot_publication_group_sizes(
        publication_audit,
        figures_dir / output_config["group_size_figure"],
    )
    plot_split_comparison(
        split_comparison,
        figures_dir / output_config["split_comparison_figure"],
    )
    plot_unseen_predicted_actual(
        model_results["final_test_predictions"],
        figures_dir / output_config["final_prediction_figure"],
    )
    if not lopo_metrics.empty:
        plot_lopo_rmse_ranking(
            lopo_metrics,
            figures_dir / output_config["lopo_ranking_figure"],
        )
        plot_lopo_bias(
            lopo_metrics,
            figures_dir / output_config["lopo_bias_figure"],
        )
        plot_shift_diagnostics(
            lopo_metrics,
            figures_dir / output_config["shift_diagnostics_figure"],
        )

    print("Week 8 publication-safe data foundation complete.")
    print("\nPublication summary:")
    print(publication_summary.round(3).to_string(index=False))
    print("\nShared publication-held-out split:")
    print(split_summary.round(3).to_string(index=False))
    print("\nLeakage audit:")
    print(leakage_audit.to_string(index=False))
    print("\nPublication-held-out model selection:")
    print(model_results["selected_summary"].round(3).to_string(index=False))
    print("\nRow-mixed versus publication-held-out:")
    print(
        split_comparison[
            [
                "split_strategy",
                "model",
                "MAE",
                "RMSE",
                "R2",
                "RMSE_gap_vs_row_mixed",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )
    if not lopo_summary.empty:
        print("\nLeave-one-publication-out summary:")
        print(lopo_summary.round(3).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the Week 8 publication-held-out data foundation."
    )
    parser.add_argument(
        "--config",
        default="configs/week08_publication_generalization.yaml",
        help="Config path relative to S1_Linear.",
    )
    args = parser.parse_args()
    main(args.config)
