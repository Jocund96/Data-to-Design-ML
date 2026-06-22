# Week 8 Strategy: Generalization to Unseen Publications

## Objective

Week 8 should test whether the Linear Family models learn transferable UHPC
material relationships or partly learn publication-specific habits.

The main evaluation unit is now a **publication**, not an individual row.
Every row from a publication must remain together during publication-held-out
evaluation.

## Prerequisite: Correct and Tune the Week 7 Baseline

Before implementing Week 8, revise the Week 7 baseline so the comparison is
fair and reusable.

Required Week 7 corrections:

1. Confirm and, if necessary, regenerate the semantic 50-percent dataset
   without dropping the first valid UHPC row.
2. Keep the Week 7 row-mixed or feature-hash-grouped train, validation, and
   test split as the baseline split strategy.
3. Put preprocessing and each Linear Family model in one complete pipeline.
4. Tune Elastic Net, Bayesian Ridge, and Polynomial Ridge with `GridSearchCV`
   using training data only. Keep OLS as the untuned reference.
5. Use group-aware cross-validation inside `GridSearchCV` so identical
   feature-hash groups remain together.
6. Select the final Week 7 model using validation performance and report the
   unchanged test set only once.
7. Freeze and save the selected model configuration so Week 8 can reuse the
   same configuration for publication-held-out comparisons.

OLS has no meaningful regularization hyperparameters, so it remains an
untuned reference baseline.

## Confirmed Data Setup

### Primary modeling dataset

Use the teammate dataset in the repository root:

```text
../semantic_recoding_features_50.csv
```

Confirmed properties:

- The corrected S1-owned semantic source has 2,073 rows and 37 columns.
- Applying the same exact-duplicate removal as Week 7 leaves 2,048 modeling
  rows across 164 represented publications.
- Contains the 28-day target `cs_28d`.
- Contains semantic-recoded mix, fiber, curing, and material features.
- Does not contain publication identifiers.
- Contains `Unnamed: 0`, which is an index artifact and must never be a model
  predictor.
- Contains both `cement_type` and `cement_type_clean`. Keep
  `cement_type_clean` as the canonical cement category and remove
  `cement_type` from the modeling features.

### Publication metadata source

Recover the missing publication metadata from:

```text
data/raw/UHPC Dataset  (Version-2).xlsx
```

Required metadata:

- `publication_id`: research paper reference in dataset
- `publication_source`: paper name, reference, or DOI text
- `publication_year`
- `publication_country`
- `mix_id`
- stable `semantic_row_id`

These columns are used only for splitting, tracking, and interpretation. They
must never enter the model feature matrix.

### Confirmed linkage method

The original workbook stores publication metadata in merged cells. Therefore:

1. Read the `UHPC Dataset ` sheet with `header=[0, 1, 2]`.
2. Flatten the multi-row column names.
3. Forward-fill only the publication metadata columns.
4. Reproduce and validate the teammate row lineage, then restore the first
   valid row accidentally removed upstream.
5. Validate that the resulting 2,073 target rows align with the corrected
   semantic dataset, allowing a small floating-point tolerance.
6. Attach the publication metadata by the validated row alignment.

The recovered metadata contains:

- 165 publications across the 2,073 corrected source rows;
- 164 publications across the 2,048 deduplicated modeling rows;
- no missing publication IDs after forward-filling;
- publication group sizes from 1 to 112 rows;
- median publication size of 8 rows.

`Mix-ID` is incomplete and not unique, so it cannot be the main row key. Use a
stable semantic row ID for predictions and worst-row tracking.

## Important Leakage Rules

1. Never use publication ID, publication source, year, country, mix ID, or row
   ID as model predictors.
2. Keep complete publications together in train, validation, and test.
3. Fit imputation, encoding, scaling, feature selection, and the model inside
   a pipeline using training publications only.
4. During hyperparameter search, use group-aware folds based on publication
   ID.
5. Never use the publication-held-out test set for model or hyperparameter
   selection.
6. During leave-one-publication-out evaluation, fit a fresh preprocessing and
   model pipeline without the held-out publication.
7. Freeze the feature configuration and hyperparameters before running the
   leave-one-publication-out experiment.

The teammate 50-percent dataset already used full-dataset missingness to
choose its upstream columns. Treat that representation as a fixed team-provided
input for the primary Week 8 analysis. All new Week 8 preprocessing and tuning
must still be training-only.

## Experiment 1: Publication Audit

Create one publication-level table before modeling.

Required fields:

- publication ID
- publication source or DOI
- country and year
- number of usable rows
- target mean, standard deviation, minimum, and maximum
- missingness summary
- dominant curing method
- fiber-used percentage
- important numeric feature ranges

Questions to answer:

- How many publications are represented?
- Are a few publications responsible for a large part of the dataset?
- Which publications have unusual strength or composition ranges?
- Which publications are too small for reliable publication-level metrics?

## Experiment 2: Shared Publication-Held-Out Split

Create one shared train, validation, and test publication split for every
Linear Family model.

Recommended target:

```text
70 percent rows: training publications
15 percent rows: validation publications
15 percent rows: test publications
```

Assign publications using publication group sizes only. Do not choose the split
based on target values. Save the final publication-to-split manifest so every
model uses exactly the same publications.

A size-balanced split is feasible at approximately:

| Split | Publications | Rows |
|---|---:|---:|
| Train | 118 | 1,450 |
| Validation | 23 | 311 |
| Test | 24 | 311 |

The final implementation must verify that publication overlap between splits
is zero.

## Experiment 3: Hyperparameter Sweep

Use the same Linear Family models as Week 7:

- OLS
- Elastic Net
- Bayesian Ridge
- Polynomial Ridge

Recommended selection procedure:

1. Build one complete preprocessing-and-model pipeline.
2. Sweep model hyperparameters using `GroupKFold` on training publications.
3. Use validation-publication RMSE and stability to select the final
   configuration.
4. Freeze the selected feature policy, preprocessing configuration, model, and
   hyperparameters.
5. Refit the frozen pipeline on train plus validation publications.
6. Report one final evaluation on unseen test publications.

Primary selection metric:

```text
validation publication-held-out RMSE
```

Also report MAE, R2, Bias, MedianAE, and publication-level variability.

## Experiment 4: Random/Row-Mixed Versus Publication-Held-Out

To show whether the previous evaluation was optimistic, compare:

- a row-mixed split where publications can appear in multiple splits;
- the publication-held-out split.

Use the same semantic dataset, features, model configuration, and frozen
hyperparameters. The only intended difference is the split strategy.

Report the generalization gap:

```text
publication-held-out metric - row-mixed metric
```

A higher publication-held-out error suggests the model was benefiting from
publication-specific laboratory practices or reporting habits.

## Experiment 5: Leave One Publication Out

Use the frozen configuration from the shared publication-held-out experiment.
Do not repeat hyperparameter search for every publication.

Primary eligibility rule:

```text
publication must contain at least 50 usable rows
```

After Week 7-equivalent exact-duplicate removal, this gives:

- 6 eligible publications;
- 446 held-out predictions;
- 21.8 percent coverage of the modeling rows.

For each eligible publication:

1. Hold out all rows from that publication.
2. Fit preprocessing and the frozen model on all other publications.
3. Predict the held-out publication.
4. Save row-level predictions.
5. Calculate publication-level metrics.

Required publication-level metrics:

- held-out row count
- MAE
- RMSE
- R2 when valid
- Bias, defined as `Actual - Predicted`
- MedianAE
- maximum absolute error

Aggregate results in two ways:

- **Micro average:** metrics across all held-out rows.
- **Macro average:** equal-weight average across publications.

The macro average prevents large publications from dominating the conclusion.

## Experiment 6: Which Publications Fail and Why?

Create a worst-publications table and a worst-rows table.

For each publication, track:

- systematic underprediction or overprediction;
- whether error is spread across the publication or caused by one extreme row;
- numeric values outside the training-publication ranges;
- unseen categorical levels;
- curing regime;
- fiber usage;
- target-strength range;
- missingness differences.

Useful diagnostic features:

- numeric out-of-training-range rate;
- unseen-category rate;
- held-out publication target mean and range;
- share of total squared error caused by the worst row;
- dominant residual direction.

Use target statistics only for post-hoc explanation, never as model inputs.

## Feature Policy Decision

The primary Week 8 analysis should keep the teammate semantic-recoded
50-percent feature policy fixed. This makes publication generalization the main
experimental variable.

An optional secondary robustness experiment may compare a stricter feature
subset, but any new missingness-based selection must be learned from training
publications only. Do not calculate a new feature policy using the held-out
publications.

## Recommended Code Structure

```text
configs/week08_publication_generalization.yaml
src/s1_linear/week08/__init__.py
src/s1_linear/week08/publication_data.py
src/s1_linear/week08/splits.py
src/s1_linear/week08/experiments.py
src/s1_linear/week08/plots.py
src/s1_linear/week08/runner.py
notebooks/week08_publication_generalization.ipynb
```

Reuse the Week 7 model, preprocessing, metrics, and plotting helpers where they
already satisfy the publication-safe workflow.

Run the Week 8 workflow as a package module:

```bash
PYTHONPATH=src python -m s1_linear.week08.runner
```

## Required Outputs

### Data and split audits

- metadata-linkage audit
- publication group summary
- publication split manifest
- split-level target and feature audit
- leakage checks showing zero publication overlap

### Model results

- group-aware hyperparameter sweep results
- row-mixed versus publication-held-out metrics
- selected frozen configuration
- final unseen-publication test predictions and metrics

### Leave-one-publication-out results

- all held-out row predictions
- one metrics row per publication
- micro and macro summaries
- worst publications
- worst rows
- publication failure explanations

### Figures

- publication group-size distribution
- row-mixed versus publication-held-out metrics
- publication RMSE/MAE ranking with row counts
- publication Bias plot
- predicted versus actual for unseen publications
- worst-publication residual plots
- error versus out-of-range/unseen-category diagnostics

## Recommended Work Order

1. Correct the semantic dataset row-lineage issue and rebuild it if required.
2. Retrofit Week 7 with leakage-safe `GridSearchCV` and freeze the tuned
   baseline configuration.
3. Build and validate the semantic-to-publication metadata linkage.
4. Create the publication audit and freeze the shared split manifest.
5. Reuse the frozen Week 7 configuration for publication-held-out evaluation.
6. Compare row-mixed and publication-held-out performance.
7. Run leave-one-publication-out evaluation.
8. Explain the worst publications and rows.
9. Create the final explanatory notebook.
