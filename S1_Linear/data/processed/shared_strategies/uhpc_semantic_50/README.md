# Shared UHPC Semantic 50% Dataset

This folder keeps the neutral shared UHPC semantic-recoded 50% dataset used by
the Linear Family work from Week 7 onward.

I have generated the local CSV files from:

`S2_Kernel/Datasets/processed/uhpc_dataset/semantic_recoding_features_50.csv`

and the publication-aware companion:

`S2_Kernel/Datasets/processed/uhpc_dataset/semantic_recoding_features_50_with_publications.csv`

The modeling-ready file drops `cement_grade`, fills missing
`fiber1_length`/`fiber1_diameter` with `0`, and keeps `cs_28d` as the target.
The publication-ready file keeps `paper_reference` for Week 8/9 splitting and
diagnostics, but that column is never used as a predictor.

Expected current row count: 2073 data rows.
