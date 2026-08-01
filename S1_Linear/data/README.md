# Data

The datasets required by the S1 Linear Family workflows are included in this
directory.

| Dataset | Path | Purpose |
|---|---|---|
| UCI Concrete | `processed/uci_concrete_clean_engineered.csv` | Baseline and representation experiments |
| UHPC workbook | `raw/UHPC Dataset  (Version-2).xlsx` | Original UHPC import |
| Target-filtered UHPC | `processed/uhpc_rows_with_28day_target.csv` | Semantic missingness experiments |
| Shared semantic UHPC | `processed/shared_strategies/uhpc_semantic_50/` | Row-mixed, publication, uncertainty, and attribution experiments |

Generated split files are retained so individual stages can be inspected.
Running the complete workflow recreates them from the included source files.
Fitted model files under `results/models/` are generated locally.
