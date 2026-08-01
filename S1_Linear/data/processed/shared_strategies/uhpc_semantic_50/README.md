# Shared UHPC Semantic 50% Dataset

This directory contains the semantic-recoded 50% missingness representation
used by the Linear Family from Week 7 onward.

The source tables were derived from:

```text
S2_Kernel/Datasets/processed/uhpc_dataset/semantic_recoding_features_50.csv
S2_Kernel/Datasets/processed/uhpc_dataset/semantic_recoding_features_50_with_publications.csv
```

The submitted copies make the S1 workflows independent of another family
directory after cloning.

| File | Contents |
|---|---|
| `uhpc_semantic_50_source.csv` | Shared source representation |
| `uhpc_semantic_50_modeling.csv` | Model inputs and `cs_28d` target |
| `uhpc_semantic_50_publication_ready.csv` | Modeling rows with publication lineage |
| `uhpc_semantic_50_with_publications.csv` | Shared publication-aware source |

The modeling table drops `cement_grade`, sets missing `fiber1_length` and
`fiber1_diameter` to zero under the shared policy, and contains 2,073 rows.
`paper_reference` is retained only for publication-aware splitting and
diagnostics; it is never a model predictor.
