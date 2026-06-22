import pandas as pd

from s1_linear.week08.publication_data import build_aligned_week08_data
from s1_linear.week08.splits import (
    make_size_balanced_publication_manifest,
    make_split_summary,
)


def test_week08_linkage_filters_modeling_and_lineage_with_same_mask():
    semantic = pd.DataFrame(
        {
            "Unnamed: 0": [0, 1, 2],
            "cement_type": ["raw", "raw", "raw"],
            "cement": [700.0, 700.0, 900.0],
            "cs_28d": [120.0, 120.0, 160.0],
        }
    )
    lineage = pd.DataFrame(
        {
            "semantic_row_id": [0, 1, 2],
            "publication_group": ["paper_a", "paper_a", "paper_b"],
        }
    )
    expected = pd.DataFrame({"cement": [700.0, 900.0], "cs_28d": [120.0, 160.0]})

    modeling, aligned_lineage, audit = build_aligned_week08_data(
        semantic_df=semantic,
        lineage_df=lineage,
        target_col="cs_28d",
        drop_predictor_columns=["Unnamed: 0", "cement_type"],
        week07_linear_ready_df=expected,
    )

    assert modeling.equals(expected)
    assert aligned_lineage["semantic_row_id"].tolist() == [0, 2]
    assert aligned_lineage["modeling_row_id"].tolist() == [0, 1]
    assert audit["exact_duplicate_rows_removed"] == 1
    assert audit["metadata_columns_in_predictors"] == 0


def test_publication_split_is_complete_and_has_zero_overlap():
    lineage = pd.DataFrame(
        {
            "publication_group": [
                "paper_a",
                "paper_a",
                "paper_a",
                "paper_b",
                "paper_b",
                "paper_c",
                "paper_c",
                "paper_d",
                "paper_e",
                "paper_f",
            ]
        }
    )
    modeling = pd.DataFrame(
        {
            "cement": range(len(lineage)),
            "cs_28d": range(100, 100 + len(lineage)),
        }
    )

    manifest = make_size_balanced_publication_manifest(
        lineage_df=lineage,
        train_size=0.6,
        validation_size=0.2,
        test_size=0.2,
        random_state=42,
        search_restarts=20,
    )
    summary, leakage = make_split_summary(
        modeling_df=modeling,
        lineage_df=lineage,
        manifest=manifest,
        target_col="cs_28d",
    )

    assert manifest["publication_group"].is_unique
    assert set(manifest["split"]) == {"train", "validation", "test"}
    assert summary["n_rows"].sum() == len(modeling)
    assert set(leakage["status"]) == {"pass"}
