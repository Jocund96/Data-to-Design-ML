"""
Restore the valid UHPC row accidentally skipped before semantic recoding.

The teammate semantic dataset is preserved as the source of truth for its
existing rows. Only the skipped first mix is reconstructed from the original
workbook using the same semantic recoding rules, then prepended to a new
S1-owned dataset. Publication metadata is saved separately and must never be
used as model input.
"""

from pathlib import Path
import argparse
import hashlib
import re
import sys

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

for path in (project_root / "src", project_root, project_root.parent):
    sys.path.insert(0, str(path))

import numpy as np
import pandas as pd

from s1_linear.config import load_config


BASE_COLUMNS = [
    "cement",
    "cement_type",
    "cement_grade",
    "silica_fume",
    "fly_ash",
    "fly_ash_type",
    "limestone_powder",
    "quartz_powder",
    "glass_powder",
    "rice_husk_ash",
    "metakaolin",
    "ggbfs",
    "slag",
    "slag_type",
    "nano_caco3",
    "nano_al2o3",
    "nano_tio2",
    "nano_sio2",
    "filler",
    "filler_type",
    "sand",
    "sand_type",
    "sand_max_size",
    "fiber1_type",
    "fiber1_amount",
    "fiber1_length",
    "fiber1_diameter",
    "fiber1_tensile_strength",
    "fiber1_youngs_modulus",
    "fiber2_type",
    "fiber2_amount",
    "fiber2_length",
    "fiber2_diameter",
    "fiber2_tensile_strength",
    "fiber2_youngs_modulus",
    "water",
    "sp_type",
    "sp_amount",
    "curing_method",
    "curing_temp",
    "curing_humidity",
    "curing_pressure",
]

AMOUNT_TYPE_PAIRS = [
    ("fly_ash", "fly_ash_type"),
    ("slag", "slag_type"),
    ("filler", "filler_type"),
    ("sand", "sand_type"),
    ("fiber1_amount", "fiber1_type"),
    ("fiber2_amount", "fiber2_type"),
    ("sp_amount", "sp_type"),
]

CEMENT_REGEX_MAP = [
    (r"high.?sulfate|type.?hs", "HS_cement"),
    (r"type.?iii|type.?3\b", "OPC_III"),
    (r"white", "white_cement"),
    (r"cem.?ii|cem2|cemii", "CEM_II"),
    (r"blast.?furnace", "BFS_cement"),
    (r"pozzolan", "pozzolan_cement"),
    (r"ggbs", "OPC_I_GGBS"),
    (r"53.?grade|grade.?53|\b53\b", "OPC_53"),
    (r"52[.,]?5", "OPC_52.5"),
    (r"42[.,]?5", "OPC_42.5"),
]

SP_REGEX_MAP = [
    (r"vma|viscosity.modif", "VMA"),
    (r"naphthalene|sulfonat", "SNF_SP"),
    (r"acrylic|acrylate|poly.acrylic.ester", "Other_Polymer_SP"),
    (r"polycarboxylic\s+ether|polycarboxylic.based|pce|hyperplastic", "PCE_HRWRA"),
    (r"polycarboxylate|polycarboxilate|carboxylate.based", "PCE_SP"),
    (r"hrwra|hrwr|high.range.water|high.performance.water", "HRWRA"),
]


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path relative to S1_Linear."""
    path = Path(path)
    return path if path.is_absolute() else (project_root / path).resolve()


def sha256_file(path: Path) -> str:
    """Return a SHA-256 hash for lineage auditing."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def categorize_fiber(value: object) -> object:
    """Apply the teammate fiber-category rules."""
    if pd.isna(value):
        return value

    text = str(value).strip()
    normalized = text.lower()
    if "hooked" in normalized:
        return "Hooked Steel Fiber"
    if "twisted" in normalized:
        return "Twisted Steel Fiber"
    if (
        "straight" in normalized
        or ("steel" in normalized and "fiber" in normalized)
        or ("steel" in normalized and "fibre" in normalized)
    ):
        return "Straight Steel Fiber"
    if normalized == "pe" or "polyethylene" in normalized:
        return "PE Fiber"
    if "pva" in normalized:
        return "PVA Fiber"
    if "glass" in normalized:
        return "Glass Fiber"
    if "carbon" in normalized:
        return "Carbon Fiber"
    if normalized == "pp":
        return "PP Fiber"
    if "cellulose" in normalized:
        return "Cellulose Fiber"
    if "wollastonit" in normalized:
        return "Wollastonite Fiber"
    if any(word in normalized for word in ("medium", "short", "long")):
        return "Straight Steel Fiber"
    return text.title()


def categorize_curing(value: object) -> object:
    """Apply the teammate curing-category rules."""
    if pd.isna(value):
        return value

    text = str(value).strip()
    normalized = text.lower()
    if "autoclave" in normalized:
        return "Autoclave Curing"
    if "steam" in normalized:
        return "Steam Curing"
    if any(term in normalized for term in ("hot water", "warm water", "warm bath")):
        return "Hot Water Curing"
    if any(term in normalized for term in ("heat", "hot", "oven")):
        return "Heat Curing"
    if "water" in normalized or "moist" in normalized:
        return "Water Curing"
    if "air" in normalized:
        return "Air Curing"
    if "standard" in normalized or "normal" in normalized:
        return "Standard Curing"
    return text.title()


def categorize_by_regex(
    value: object,
    regex_map: list[tuple[str, str]],
    default: str,
) -> object:
    """Map a reported category using the teammate regex ordering."""
    if pd.isna(value) or value in ("unknown_type", "not_applicable"):
        return value

    normalized = str(value).lower()
    return next(
        (label for pattern, label in regex_map if re.search(pattern, normalized)),
        default,
    )


def clean_cement_combined(cement_type: object, cement_grade: object) -> str:
    """Combine standardized cement type and grade as the teammate did."""
    cement = str(cement_type).strip() if pd.notna(cement_type) else ""
    grade = str(cement_grade).strip() if pd.notna(cement_grade) else ""

    special_types = {
        "HS_cement",
        "OPC_III",
        "CEM_II",
        "white_cement",
        "pozzolan_cement",
        "OPC_I_GGBS",
        "BFS_cement",
    }
    if cement in special_types:
        return cement
    if grade in ("52.5", "52.50"):
        return "OPC_52.5"
    if grade in ("42.5", "42.50"):
        return "OPC_42.5"
    if grade in ("53.0", "53"):
        return "OPC_53"
    if cement in ("OPC_52.5", "OPC_42.5", "OPC_53"):
        return cement
    if cement == "OPC_unknown":
        return "OPC_I"
    return "Unknown"


def recode_semantic_missingness(row: pd.Series) -> pd.Series:
    """Distinguish unknown material type from a material not being used."""
    row = row.copy()
    for amount_col, type_col in AMOUNT_TYPE_PAIRS:
        amount = pd.to_numeric(row[amount_col], errors="coerce")
        type_missing = pd.isna(row[type_col])

        if pd.notna(amount) and amount > 0 and type_missing:
            row[type_col] = "unknown_type"
        elif (pd.isna(amount) or amount == 0) and type_missing:
            row[type_col] = "not_applicable"
            row[amount_col] = 0.0

    return row


def load_original_valid_rows(
    workbook_path: Path,
    sheet_name: str,
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load original features and separate publication lineage for valid targets."""
    original = pd.read_excel(workbook_path, sheet_name=sheet_name, header=[0, 2])

    # The first 42 selected columns are mix/curing inputs; original column 53 is cs_28d.
    selected_positions = list(range(1, 43)) + [53]
    feature_target = original.iloc[:, selected_positions].copy()
    feature_target.columns = BASE_COLUMNS + [target_col]
    feature_target[target_col] = pd.to_numeric(feature_target[target_col], errors="coerce")
    valid_mask = feature_target[target_col].notna()

    lineage = original.iloc[:, [0, 88, 89, 90, 91]].copy()
    lineage.columns = [
        "mix_id",
        "publication_country",
        "publication_source",
        "publication_year",
        "publication_reference_id",
    ]
    metadata_columns = [
        "publication_country",
        "publication_source",
        "publication_year",
        "publication_reference_id",
    ]
    lineage[metadata_columns] = lineage[metadata_columns].ffill()

    return (
        feature_target.loc[valid_mask].reset_index(drop=True),
        lineage.loc[valid_mask].reset_index(drop=True),
    )


def build_restored_semantic_row(
    original_row: pd.Series,
    source_columns: list[str],
    target_col: str,
) -> pd.DataFrame:
    """Recode the skipped original row into the existing semantic schema."""
    row = original_row.copy()

    row["fly_ash_type"] = (
        str(row["fly_ash_type"]).strip()[-1] if pd.notna(row["fly_ash_type"]) else np.nan
    )
    row["fly_ash_type"] = {"F": "class F", "C": "class C"}.get(
        row["fly_ash_type"],
        row["fly_ash_type"],
    )
    for column in ("slag_type", "filler_type", "sand_type"):
        row[column] = str(row[column]).strip().title() if pd.notna(row[column]) else np.nan

    row["fiber1_type"] = categorize_fiber(row["fiber1_type"])
    row["fiber2_type"] = categorize_fiber(row["fiber2_type"])
    row["curing_method"] = categorize_curing(row["curing_method"])
    row["curing_method"] = {
        "Curing  At 90 Oc": "Heat Curing",
        "Stnadrad Curing": "Standard Curing",
    }.get(row["curing_method"], row["curing_method"])

    row["cement_type"] = categorize_by_regex(
        row["cement_type"],
        CEMENT_REGEX_MAP,
        default="OPC_unknown",
    )
    row["cement_type_clean"] = clean_cement_combined(
        row["cement_type"],
        row["cement_grade"],
    )
    row["sp_type"] = categorize_by_regex(
        row["sp_type"],
        SP_REGEX_MAP,
        default="Unspecified",
    )
    row = recode_semantic_missingness(row)
    row[target_col] = pd.to_numeric(row[target_col], errors="coerce")

    restored = pd.DataFrame([row])
    restored.insert(0, "Unnamed: 0", 0)

    missing_columns = [column for column in source_columns if column not in restored]
    if missing_columns:
        raise ValueError(f"Cannot reconstruct source columns: {missing_columns}")

    return restored.loc[:, source_columns]


def validate_and_correct(
    semantic_source: pd.DataFrame,
    original_valid: pd.DataFrame,
    target_col: str,
    expected_source_rows: int,
    expected_corrected_rows: int,
    target_tolerance: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Validate row lineage and prepend the reconstructed skipped row."""
    if len(semantic_source) != expected_source_rows:
        raise ValueError(
            f"Expected {expected_source_rows} semantic rows, found {len(semantic_source)}."
        )
    if len(original_valid) != expected_corrected_rows:
        raise ValueError(
            f"Expected {expected_corrected_rows} valid original rows, "
            f"found {len(original_valid)}."
        )
    if target_col not in semantic_source:
        raise ValueError(f"Target column is missing from semantic source: {target_col}")

    source_targets = pd.to_numeric(semantic_source[target_col], errors="coerce").to_numpy()
    expected_targets = original_valid[target_col].iloc[1:].to_numpy()
    target_differences = np.abs(source_targets - expected_targets)
    sequence_matches = bool(np.all(target_differences <= target_tolerance))
    if not sequence_matches:
        raise ValueError(
            "Semantic rows do not align with original valid rows after the first row."
        )

    restored = build_restored_semantic_row(
        original_row=original_valid.iloc[0],
        source_columns=semantic_source.columns.tolist(),
        target_col=target_col,
    )
    corrected = pd.concat([restored, semantic_source], ignore_index=True)
    if "Unnamed: 0" in corrected:
        corrected["Unnamed: 0"] = np.arange(len(corrected))

    if len(corrected) != expected_corrected_rows:
        raise ValueError("Corrected dataset row count does not match expectation.")
    if corrected[target_col].isna().any():
        raise ValueError("Corrected dataset contains missing target values.")

    unchanged_columns = [column for column in semantic_source if column != "Unnamed: 0"]
    suffix_unchanged = corrected.iloc[1:][unchanged_columns].reset_index(drop=True).equals(
        semantic_source[unchanged_columns].reset_index(drop=True)
    )
    if not suffix_unchanged:
        raise ValueError("Existing semantic rows changed during correction.")

    audit = {
        "source_rows": len(semantic_source),
        "original_valid_target_rows": len(original_valid),
        "corrected_rows": len(corrected),
        "restored_rows": len(corrected) - len(semantic_source),
        "restored_target": corrected.loc[0, target_col],
        "source_target_sequence_matches_original_after_skipped_row": sequence_matches,
        "max_target_sequence_absolute_difference": float(target_differences.max()),
        "existing_semantic_rows_unchanged": suffix_unchanged,
        "schema_unchanged": corrected.columns.tolist() == semantic_source.columns.tolist(),
        "missing_targets_after_correction": int(corrected[target_col].isna().sum()),
    }
    return corrected, audit


def make_audit_table(
    audit: dict[str, object],
    paths: dict[str, Path],
    restored_mix_id: object,
) -> pd.DataFrame:
    """Create a compact correction and lineage audit table."""
    path_rows = {
        "semantic_source_path": paths["semantic_source"],
        "semantic_source_sha256": sha256_file(paths["semantic_source"]),
        "original_workbook_path": paths["original_workbook"],
        "original_workbook_sha256": sha256_file(paths["original_workbook"]),
        "corrected_dataset_path": paths["corrected_dataset"],
        "corrected_dataset_sha256": sha256_file(paths["corrected_dataset"]),
        "lineage_path": paths["lineage"],
        "lineage_sha256": sha256_file(paths["lineage"]),
        "restored_mix_id": restored_mix_id,
        "lineage_is_for_splitting_and_reporting_only": True,
    }
    return pd.DataFrame(
        [{"item": key, "value": value} for key, value in {**path_rows, **audit}.items()]
    )


def main(config_path: str) -> None:
    """Build and audit the corrected 2,073-row semantic dataset."""
    config = load_config(resolve_project_path(config_path))
    data_config = config["data"]
    output_config = config["outputs"]

    paths = {
        "semantic_source": resolve_project_path(data_config["semantic_source_path"]),
        "original_workbook": resolve_project_path(data_config["original_workbook_path"]),
        "corrected_dataset": resolve_project_path(output_config["corrected_dataset_path"]),
        "lineage": resolve_project_path(output_config["lineage_path"]),
        "audit": resolve_project_path(output_config["audit_path"]),
    }
    for required_path in (paths["semantic_source"], paths["original_workbook"]):
        if not required_path.exists():
            raise FileNotFoundError(f"Required source not found: {required_path}")

    semantic_source = pd.read_csv(paths["semantic_source"])
    original_valid, lineage = load_original_valid_rows(
        workbook_path=paths["original_workbook"],
        sheet_name=data_config["original_sheet_name"],
        target_col=data_config["target"],
    )
    corrected, audit = validate_and_correct(
        semantic_source=semantic_source,
        original_valid=original_valid,
        target_col=data_config["target"],
        expected_source_rows=data_config["expected_source_rows"],
        expected_corrected_rows=data_config["expected_corrected_rows"],
        target_tolerance=data_config["target_sequence_tolerance"],
    )

    lineage.insert(0, "semantic_row_id", np.arange(len(lineage)))
    lineage["publication_group"] = lineage["publication_reference_id"].fillna(
        lineage["publication_source"]
    )

    for output_path in (paths["corrected_dataset"], paths["lineage"], paths["audit"]):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    corrected.to_csv(paths["corrected_dataset"], index=False)
    lineage.to_csv(paths["lineage"], index=False)

    audit_table = make_audit_table(
        audit=audit,
        paths=paths,
        restored_mix_id=lineage.loc[0, "mix_id"],
    )
    audit_table.to_csv(paths["audit"], index=False)

    print("Corrected semantic dataset created.")
    print(audit_table.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Restore the semantic dataset row skipped upstream."
    )
    parser.add_argument(
        "--config",
        default="configs/week07_correct_semantic_dataset.yaml",
        help="Config path relative to S1_Linear by default.",
    )
    args = parser.parse_args()
    main(args.config)
