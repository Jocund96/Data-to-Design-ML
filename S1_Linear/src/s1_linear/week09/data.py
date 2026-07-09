"""Load and validate the frozen Week 8 inputs required by Week 9."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from s1_linear.config import load_config
from s1_linear.week08.experiments import load_split


LINEAGE_ONLY_COLUMNS = {
    "semantic_row_id",
    "modeling_row_id",
    "mix_id",
    "publication_country",
    "publication_source",
    "publication_year",
    "publication_reference_id",
    "publication_group",
}


@dataclass
class Week09Inputs:
    """Validated Week 8 data and configuration reused in Week 9."""

    X_train: pd.DataFrame
    y_train: pd.Series
    train_lineage: pd.DataFrame
    X_calibration: pd.DataFrame
    y_calibration: pd.Series
    calibration_lineage: pd.DataFrame
    X_test: pd.DataFrame
    y_test: pd.Series
    test_lineage: pd.DataFrame
    modeling_df: pd.DataFrame
    modeling_lineage: pd.DataFrame
    split_manifest: pd.DataFrame
    frozen_config: dict
    selected_model_name: str
    minimum_lopo_rows: int
    target_col: str
    policy: str


def _audit_row(check: str, value, passed: bool, detail: str = "") -> dict:
    return {
        "check": check,
        "value": value,
        "status": "pass" if passed else "fail",
        "detail": detail,
    }


def _publication_overlap_audit(lineages: dict[str, pd.DataFrame]) -> list[dict]:
    groups = {
        name: set(frame["publication_group"].astype(str))
        for name, frame in lineages.items()
    }
    rows = []
    names = list(groups)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            overlap = groups[first] & groups[second]
            rows.append(
                _audit_row(
                    f"{first}_{second}_publication_overlap",
                    len(overlap),
                    not overlap,
                    ", ".join(sorted(overlap)[:5]),
                )
            )
    return rows


def load_week09_inputs(config: dict, project_root: Path) -> tuple[Week09Inputs, pd.DataFrame]:
    """Load Week 8 artifacts and fail if a Week 9 prerequisite is unsafe."""
    data_config = config["data"]

    def resolve(path: str) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else (project_root / candidate).resolve()

    required_paths = {
        "week08_config": resolve(data_config["week08_config_path"]),
        "week08_split_dir": resolve(data_config["week08_split_dir"]),
        "week08_modeling_data": resolve(data_config["week08_modeling_data_path"]),
        "week08_modeling_lineage": resolve(
            data_config["week08_modeling_lineage_path"]
        ),
        "week08_split_manifest": resolve(data_config["week08_split_manifest_path"]),
        "week08_frozen_config": resolve(data_config["week08_frozen_config_path"]),
        "week08_lopo_metrics": resolve(data_config["week08_lopo_metrics_path"]),
    }
    audit_rows = [
        _audit_row(f"{name}_exists", path.exists(), path.exists(), str(path))
        for name, path in required_paths.items()
    ]
    missing = [name for name, path in required_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required Week 8 artifacts: {missing}")

    target_col = data_config["target"]
    split_dir = required_paths["week08_split_dir"]
    X_train, y_train, train_lineage = load_split(split_dir, "train", target_col)
    X_calibration, y_calibration, calibration_lineage = load_split(
        split_dir,
        "validation",
        target_col,
    )
    X_test, y_test, test_lineage = load_split(split_dir, "test", target_col)
    modeling_df = pd.read_csv(required_paths["week08_modeling_data"])
    modeling_lineage = pd.read_csv(required_paths["week08_modeling_lineage"])
    split_manifest = pd.read_csv(required_paths["week08_split_manifest"])
    frozen_config = load_config(required_paths["week08_frozen_config"])
    week08_config = load_config(required_paths["week08_config"])

    split_frames = {
        "train": (X_train, y_train, train_lineage),
        "calibration": (X_calibration, y_calibration, calibration_lineage),
        "test": (X_test, y_test, test_lineage),
    }
    for name, (X, y, lineage) in split_frames.items():
        aligned = len(X) == len(y) == len(lineage)
        audit_rows.append(
            _audit_row(
                f"{name}_row_alignment",
                f"{len(X)}/{len(y)}/{len(lineage)}",
                aligned,
                "X/y/lineage row counts",
            )
        )

    same_features = list(X_train.columns) == list(X_calibration.columns) == list(
        X_test.columns
    )
    audit_rows.append(
        _audit_row(
            "shared_predictor_schema",
            len(X_train.columns),
            same_features,
            "train, calibration, and test columns must match in order",
        )
    )
    metadata_predictors = sorted(LINEAGE_ONLY_COLUMNS & set(X_train.columns))
    audit_rows.append(
        _audit_row(
            "lineage_columns_excluded_from_predictors",
            len(metadata_predictors),
            not metadata_predictors,
            ", ".join(metadata_predictors),
        )
    )
    audit_rows.extend(
        _publication_overlap_audit(
            {
                "train": train_lineage,
                "calibration": calibration_lineage,
                "test": test_lineage,
            }
        )
    )

    total_split_rows = sum(len(values[0]) for values in split_frames.values())
    audit_rows.append(
        _audit_row(
            "all_modeling_rows_represented_in_shared_split",
            total_split_rows,
            total_split_rows == len(modeling_df) == len(modeling_lineage),
            f"modeling data/lineage rows = {len(modeling_df)}/{len(modeling_lineage)}",
        )
    )
    target_present = target_col in modeling_df and target_col not in X_train
    audit_rows.append(
        _audit_row(
            "target_separated_from_predictors",
            target_col,
            target_present,
        )
    )
    manifest_complete = (
        split_manifest["publication_group"].is_unique
        and set(split_manifest["split"]) == {"train", "validation", "test"}
        and not split_manifest["assignment_uses_target_values"].astype(bool).any()
    )
    audit_rows.append(
        _audit_row(
            "week08_manifest_is_target_independent",
            len(split_manifest),
            manifest_complete,
        )
    )

    enabled_models = frozen_config.get("enabled_models", [])
    selected_model_name = enabled_models[0] if len(enabled_models) == 1 else ""
    frozen = (
        frozen_config.get("hyperparameter_tuning", {}).get("enabled") is False
        and bool(
            frozen_config.get("hyperparameter_tuning", {}).get(
                "frozen_from_training_cv"
            )
        )
        and selected_model_name == "Elastic Net"
    )
    audit_rows.append(
        _audit_row(
            "week08_selected_configuration_is_frozen",
            selected_model_name,
            frozen,
        )
    )
    bayesian_available = "bayesian_ridge" in frozen_config.get("models", {})
    audit_rows.append(
        _audit_row(
            "week08_bayesian_ridge_parameters_available",
            bayesian_available,
            bayesian_available,
        )
    )

    minimum_lopo_rows = int(
        week08_config["publication_audit"]["minimum_rows_for_reliable_metrics"]
    )
    audit_rows.append(
        _audit_row(
            "week08_lopo_threshold_loaded",
            minimum_lopo_rows,
            minimum_lopo_rows == 50,
            "Week 9 does not maintain an independent threshold",
        )
    )

    audit = pd.DataFrame(audit_rows)
    failures = audit.loc[audit["status"].eq("fail"), "check"].tolist()
    if failures:
        raise ValueError(f"Week 9 input readiness audit failed: {failures}")

    return (
        Week09Inputs(
            X_train=X_train,
            y_train=y_train,
            train_lineage=train_lineage,
            X_calibration=X_calibration,
            y_calibration=y_calibration,
            calibration_lineage=calibration_lineage,
            X_test=X_test,
            y_test=y_test,
            test_lineage=test_lineage,
            modeling_df=modeling_df,
            modeling_lineage=modeling_lineage,
            split_manifest=split_manifest,
            frozen_config=frozen_config,
            selected_model_name=selected_model_name,
            minimum_lopo_rows=minimum_lopo_rows,
            target_col=target_col,
            policy=data_config["policy"],
        ),
        audit,
    )

