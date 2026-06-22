"""Publication-held-out split utilities for Week 8."""

from pathlib import Path

import numpy as np
import pandas as pd


SPLIT_NAMES = ("train", "validation", "test")


def _split_objective(
    row_counts: dict[str, int],
    target_rows: dict[str, float],
) -> float:
    """Measure squared proportional distance from target split sizes."""
    return float(
        sum(
            ((row_counts[name] - target_rows[name]) / target_rows[name]) ** 2
            for name in SPLIT_NAMES
        )
    )


def _largest_remainder_counts(
    total_count: int,
    fractions: dict[str, float],
) -> dict[str, int]:
    """Convert fractional publication targets into counts that sum exactly."""
    raw_targets = {name: total_count * fractions[name] for name in SPLIT_NAMES}
    targets = {name: int(np.floor(raw_targets[name])) for name in SPLIT_NAMES}
    remaining = total_count - sum(targets.values())
    remainder_order = sorted(
        SPLIT_NAMES,
        key=lambda name: raw_targets[name] - targets[name],
        reverse=True,
    )
    for name in remainder_order[:remaining]:
        targets[name] += 1
    return targets


def _improve_assignment_with_swaps(
    assignments: dict[str, str],
    group_sizes: dict[str, int],
    row_counts: dict[str, int],
    target_rows: dict[str, float],
) -> tuple[dict[str, str], dict[str, int]]:
    """Swap publications between splits until no row-balance improvement remains."""
    assignments = assignments.copy()
    row_counts = row_counts.copy()

    while True:
        current_score = _split_objective(row_counts, target_rows)
        best_score = current_score
        best_swap = None
        groups = list(assignments)

        for index, first_group in enumerate(groups):
            first_split = assignments[first_group]
            for second_group in groups[index + 1 :]:
                second_split = assignments[second_group]
                if first_split == second_split:
                    continue

                candidate_counts = row_counts.copy()
                candidate_counts[first_split] += (
                    group_sizes[second_group] - group_sizes[first_group]
                )
                candidate_counts[second_split] += (
                    group_sizes[first_group] - group_sizes[second_group]
                )
                score = _split_objective(candidate_counts, target_rows)
                if score < best_score - 1e-15:
                    best_score = score
                    best_swap = (first_group, second_group, candidate_counts)

        if best_swap is None:
            return assignments, row_counts

        first_group, second_group, row_counts = best_swap
        assignments[first_group], assignments[second_group] = (
            assignments[second_group],
            assignments[first_group],
        )


def make_size_balanced_publication_manifest(
    lineage_df: pd.DataFrame,
    train_size: float,
    validation_size: float,
    test_size: float,
    random_state: int,
    search_restarts: int = 500,
) -> pd.DataFrame:
    """
    Assign complete publications using group sizes only.

    Publication counts are fixed near the requested proportions. Multiple
    randomized greedy passes and publication swaps then minimize row imbalance.
    Target values are never used.
    """
    fractions = {
        "train": train_size,
        "validation": validation_size,
        "test": test_size,
    }
    if not np.isclose(sum(fractions.values()), 1.0):
        raise ValueError("Publication split fractions must sum to 1.")
    if lineage_df["publication_group"].isna().any():
        raise ValueError("Publication groups must be complete before splitting.")

    counts = lineage_df["publication_group"].value_counts().rename_axis(
        "publication_group"
    ).reset_index(name="n_rows")
    target_rows = {name: fraction * len(lineage_df) for name, fraction in fractions.items()}
    target_publications = _largest_remainder_counts(len(counts), fractions)
    group_sizes = counts.set_index("publication_group")["n_rows"].to_dict()
    rng = np.random.default_rng(random_state)
    best_assignment = None
    best_score = np.inf

    for _ in range(search_restarts):
        candidates = counts.copy()
        candidates["priority"] = (
            candidates["n_rows"] + rng.uniform(0, candidates["n_rows"].max() * 0.15, len(candidates))
        )
        candidates = candidates.sort_values("priority", ascending=False)
        row_counts = {name: 0 for name in SPLIT_NAMES}
        publication_counts = {name: 0 for name in SPLIT_NAMES}
        assignments = {}

        for _, publication in candidates.iterrows():
            split_order = [
                name
                for name in rng.permutation(SPLIT_NAMES)
                if publication_counts[name] < target_publications[name]
            ]
            chosen_split = min(
                split_order,
                key=lambda name: _split_objective(
                    {
                        **row_counts,
                        name: row_counts[name] + int(publication["n_rows"]),
                    },
                    target_rows,
                ),
            )
            assignments[publication["publication_group"]] = chosen_split
            row_counts[chosen_split] += int(publication["n_rows"])
            publication_counts[chosen_split] += 1

        assignments, row_counts = _improve_assignment_with_swaps(
            assignments=assignments,
            group_sizes=group_sizes,
            row_counts=row_counts,
            target_rows=target_rows,
        )
        score = _split_objective(row_counts, target_rows)
        if score < best_score and all(row_counts[name] > 0 for name in SPLIT_NAMES):
            best_score = score
            best_assignment = assignments

    if best_assignment is None:
        raise RuntimeError("Could not create a valid publication-held-out split.")

    manifest = counts.copy()
    manifest["split"] = manifest["publication_group"].map(best_assignment)
    manifest["target_row_count"] = manifest["split"].map(target_rows)
    manifest["target_publication_count"] = manifest["split"].map(target_publications)
    manifest["assignment_uses_target_values"] = False
    return manifest.sort_values(["split", "n_rows", "publication_group"], ascending=[True, False, True])


def make_split_summary(
    modeling_df: pd.DataFrame,
    lineage_df: pd.DataFrame,
    manifest: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create split statistics and publication-overlap checks."""
    split_map = manifest.set_index("publication_group")["split"]
    row_splits = lineage_df["publication_group"].map(split_map)
    if row_splits.isna().any():
        raise ValueError("Some modeling rows were not assigned to a publication split.")

    rows = []
    for split_name in SPLIT_NAMES:
        mask = row_splits.eq(split_name)
        y = pd.to_numeric(modeling_df.loc[mask, target_col], errors="coerce")
        X = modeling_df.loc[mask].drop(columns=target_col)
        rows.append(
            {
                "split": split_name,
                "n_rows": int(mask.sum()),
                "row_percentage": float(mask.mean() * 100),
                "n_publications": int(lineage_df.loc[mask, "publication_group"].nunique()),
                "target_mean": y.mean(),
                "target_std": y.std(),
                "target_min": y.min(),
                "target_max": y.max(),
                "missing_predictor_cells": int(X.isna().sum().sum()),
                "missing_predictor_cells_per_row": float(X.isna().sum(axis=1).mean()),
            }
        )

    split_groups = {
        split_name: set(
            manifest.loc[manifest["split"].eq(split_name), "publication_group"]
        )
        for split_name in SPLIT_NAMES
    }
    overlaps = {
        "train_validation_publication_overlap": len(
            split_groups["train"] & split_groups["validation"]
        ),
        "train_test_publication_overlap": len(
            split_groups["train"] & split_groups["test"]
        ),
        "validation_test_publication_overlap": len(
            split_groups["validation"] & split_groups["test"]
        ),
    }
    leakage_audit = pd.DataFrame(
        [
            {"check": name, "value": value, "status": "pass" if value == 0 else "fail"}
            for name, value in overlaps.items()
        ]
        + [
            {
                "check": "manifest_assignment_uses_target_values",
                "value": bool(manifest["assignment_uses_target_values"].any()),
                "status": (
                    "pass"
                    if not manifest["assignment_uses_target_values"].any()
                    else "fail"
                ),
            },
            {
                "check": "all_modeling_rows_assigned",
                "value": int(row_splits.notna().sum()),
                "status": "pass" if row_splits.notna().all() else "fail",
            },
        ]
    )
    return pd.DataFrame(rows), leakage_audit


def save_publication_splits(
    modeling_df: pd.DataFrame,
    lineage_df: pd.DataFrame,
    manifest: pd.DataFrame,
    target_col: str,
    split_dir: Path,
) -> None:
    """Save aligned predictor, target, and lineage files for every split."""
    split_dir.mkdir(parents=True, exist_ok=True)
    split_map = manifest.set_index("publication_group")["split"]
    row_splits = lineage_df["publication_group"].map(split_map)

    for split_name in SPLIT_NAMES:
        mask = row_splits.eq(split_name)
        split_modeling = modeling_df.loc[mask].reset_index(drop=True)
        split_lineage = lineage_df.loc[mask].reset_index(drop=True)
        split_modeling.drop(columns=target_col).to_csv(
            split_dir / f"X_{split_name}.csv",
            index=False,
        )
        split_modeling[[target_col]].to_csv(
            split_dir / f"y_{split_name}.csv",
            index=False,
        )
        split_lineage.to_csv(
            split_dir / f"lineage_{split_name}.csv",
            index=False,
        )
