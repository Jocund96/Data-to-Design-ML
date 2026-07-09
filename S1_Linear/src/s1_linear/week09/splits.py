"""Train/calibration/held-out publication role utilities for Week 9."""

import numpy as np
import pandas as pd


ROLE_NAMES = ("train", "calibration", "heldout")


def make_shared_role_manifest(week08_manifest: pd.DataFrame) -> pd.DataFrame:
    """Translate the frozen Week 8 split manifest into Week 9 role names."""
    role_map = {
        "train": "train",
        "validation": "calibration",
        "test": "heldout",
    }
    manifest = week08_manifest.copy()
    manifest["week09_role"] = manifest["split"].map(role_map)
    manifest.insert(0, "evaluation_fold", "shared_publication_test")
    if manifest["week09_role"].isna().any():
        raise ValueError("Week 8 manifest contains an unsupported split name.")
    return manifest


def eligible_lopo_publications(
    lineage_df: pd.DataFrame,
    minimum_rows: int,
) -> pd.DataFrame:
    """Return publication groups meeting the frozen Week 8 reliability threshold."""
    counts = (
        lineage_df["publication_group"]
        .value_counts()
        .rename_axis("publication_group")
        .reset_index(name="n_rows")
    )
    return counts.loc[counts["n_rows"].ge(minimum_rows)].sort_values(
        ["n_rows", "publication_group"],
        ascending=[False, True],
    )


def make_lopo_role_manifest(
    lineage_df: pd.DataFrame,
    week08_manifest: pd.DataFrame,
    heldout_publication: str,
    calibration_manifest_split: str = "validation",
) -> pd.DataFrame:
    """Assign one LOPO publication and a target-independent calibration role."""
    counts = (
        lineage_df["publication_group"]
        .value_counts()
        .rename_axis("publication_group")
        .reset_index(name="n_rows")
    )
    week08_split = week08_manifest.set_index("publication_group")["split"]
    counts["week08_split"] = counts["publication_group"].map(week08_split)
    if counts["week08_split"].isna().any():
        raise ValueError("LOPO manifest contains publications absent from Week 8 manifest.")
    if heldout_publication not in set(counts["publication_group"]):
        raise ValueError(f"Unknown held-out publication: {heldout_publication}")

    counts["week09_role"] = np.select(
        [
            counts["publication_group"].eq(heldout_publication),
            counts["week08_split"].eq(calibration_manifest_split),
        ],
        ["heldout", "calibration"],
        default="train",
    )
    counts.insert(0, "evaluation_fold", heldout_publication)
    counts["assignment_uses_target_values"] = False
    return counts


def split_modeling_data_by_role(
    modeling_df: pd.DataFrame,
    lineage_df: pd.DataFrame,
    role_manifest: pd.DataFrame,
    target_col: str,
) -> dict[str, tuple[pd.DataFrame, pd.Series, pd.DataFrame]]:
    """Create aligned train, calibration, and held-out frames from a role manifest."""
    role_map = role_manifest.set_index("publication_group")["week09_role"]
    row_roles = lineage_df["publication_group"].map(role_map)
    if row_roles.isna().any():
        raise ValueError("Some modeling rows are missing from the Week 9 role manifest.")

    result = {}
    for role in ROLE_NAMES:
        mask = row_roles.eq(role)
        X = modeling_df.loc[mask].drop(columns=target_col).reset_index(drop=True)
        y = modeling_df.loc[mask, target_col].reset_index(drop=True)
        lineage = lineage_df.loc[mask].reset_index(drop=True)
        if not len(X):
            raise ValueError(f"Week 9 role '{role}' has no rows.")
        result[role] = (X, y, lineage)
    return result


def audit_role_manifest(
    lineage_df: pd.DataFrame,
    role_manifest: pd.DataFrame,
    expected_heldout_publication: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize one role manifest and verify publication-level separation."""
    role_map = role_manifest.set_index("publication_group")["week09_role"]
    row_roles = lineage_df["publication_group"].map(role_map)
    fold_name = str(role_manifest["evaluation_fold"].iloc[0])
    summary_rows = []
    group_sets = {}

    for role in ROLE_NAMES:
        groups = set(
            role_manifest.loc[
                role_manifest["week09_role"].eq(role),
                "publication_group",
            ].astype(str)
        )
        group_sets[role] = groups
        summary_rows.append(
            {
                "evaluation_fold": fold_name,
                "role": role,
                "n_rows": int(row_roles.eq(role).sum()),
                "n_publications": len(groups),
            }
        )

    audit_rows = []
    for index, first in enumerate(ROLE_NAMES):
        for second in ROLE_NAMES[index + 1 :]:
            overlap = group_sets[first] & group_sets[second]
            audit_rows.append(
                {
                    "evaluation_fold": fold_name,
                    "check": f"{first}_{second}_publication_overlap",
                    "value": len(overlap),
                    "status": "pass" if not overlap else "fail",
                }
            )
    audit_rows.extend(
        [
            {
                "evaluation_fold": fold_name,
                "check": "all_rows_assigned",
                "value": int(row_roles.notna().sum()),
                "status": "pass" if row_roles.notna().all() else "fail",
            },
            {
                "evaluation_fold": fold_name,
                "check": "assignment_uses_target_values",
                "value": bool(
                    role_manifest.get(
                        "assignment_uses_target_values",
                        pd.Series(False, index=role_manifest.index),
                    )
                    .astype(bool)
                    .any()
                ),
                "status": (
                    "pass"
                    if not role_manifest.get(
                        "assignment_uses_target_values",
                        pd.Series(False, index=role_manifest.index),
                    )
                    .astype(bool)
                    .any()
                    else "fail"
                ),
            },
        ]
    )
    if expected_heldout_publication is not None:
        heldout_groups = group_sets["heldout"]
        correct = heldout_groups == {expected_heldout_publication}
        audit_rows.append(
            {
                "evaluation_fold": fold_name,
                "check": "expected_heldout_publication_only",
                "value": ", ".join(sorted(heldout_groups)),
                "status": "pass" if correct else "fail",
            }
        )

    summary = pd.DataFrame(summary_rows)
    audit = pd.DataFrame(audit_rows)
    failures = audit.loc[audit["status"].eq("fail")]
    if not failures.empty:
        raise ValueError(
            f"Week 9 role-manifest audit failed for {fold_name}: "
            f"{failures['check'].tolist()}"
        )
    return summary, audit

