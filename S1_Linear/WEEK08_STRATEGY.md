# Week 8 Strategy: Publication Generalization

## Objective

Week 8 tests whether the Linear Family UHPC models generalize to publications
that were completely unseen during training. The evaluation unit is therefore a
publication group, not an individual row.

## Data Setup

Week 8 uses the corrected shared semantic-recoded 50 percent dataset from:

```text
data/processed/shared_strategies/uhpc_semantic_50/
```

The active files are:

- `uhpc_semantic_50_modeling.csv`: 2,073 modeling rows, 33 predictors plus
  `cs_28d`;
- `uhpc_semantic_50_publication_ready.csv`: the same 2,073 rows plus
  `paper_reference` for publication grouping.

Publication metadata is used only for splitting, lineage, diagnostics, and
post-hoc interpretation. It is removed from the model predictors.

## Leakage Rules

1. Keep complete publications together in train, validation, and test.
2. Do not use `paper_reference` or any lineage field as a model predictor.
3. Fit imputation, encoding, scaling, and models inside training-only
   pipelines.
4. Use publication-group cross-validation during hyperparameter tuning.
5. Select the final model on validation publications only.
6. Evaluate the frozen selected model once on unseen test publications.

## Experiments

1. Build the publication audit and identify publications with enough rows for
   reliable publication-level metrics.
2. Create a shared 70/15/15 publication-held-out split.
3. Tune the four Linear Family models: OLS, Elastic Net, Bayesian Ridge, and
   Polynomial Ridge.
4. Compare the same frozen model on the Week 7 row-mixed split versus the Week
   8 publication-held-out split.
5. Run leave-one-publication-out evaluation for publications with at least 50
   rows.
6. Diagnose the worst publications and worst rows using post-hoc feature,
   target, missingness, fiber, curing, and residual summaries.

## Current Result Anchor

The corrected Week 8 run uses all 2,073 shared rows across 165 publication
groups. Six publications meet the 50-row reliability threshold. The selected
publication-held-out model is Elastic Net.

The final test comparison shows a clear generalization gap:

- row-mixed RMSE: 21.214;
- unseen-publication RMSE: 30.218;
- gap: +9.004 RMSE.

This supports the main Week 8 interpretation: row-mixed evaluation is
optimistic for this UHPC dataset because it can benefit from publication-level
reporting and laboratory patterns that do not transfer cleanly to unseen
publications.

## Run Command

From `S1_Linear/`:

```bash
PYTHONPATH=src python -m s1_linear.week08.runner
```
