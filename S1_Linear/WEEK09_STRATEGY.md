# Week 9 Strategy: Uncertainty and Calibration

## Objective

Week 9 asks a different question from Week 8:

> Given a UHPC strength prediction for a publication that was not used to fit
> the model, how much should that prediction be trusted?

The primary output is no longer only a point prediction. Each prediction must
also have a prediction interval or uncertainty score, followed by an honest
evaluation of interval coverage and width.

The Week 9 work must reuse the Week 8 publication-safe dataset, split manifest,
selected model configuration, lineage, and thresholded leave-one-publication-out
(LOPO) definition. It must not retune the point-prediction model.

## Requirements From the Week 9 Brief

The brief requires us to:

1. Start from the Week 8 selected configurations.
2. Use validation data to estimate interval size.
3. Fit preprocessing and models using training rows only.
4. Keep publication IDs out of model predictors.
5. Generate prediction intervals or uncertainty scores.
6. Report empirical coverage and interval width.
7. Compare the shared publication-held-out evaluation with thresholded LOPO.
8. Identify overconfident and underconfident predictions or publications.
9. Use split conformal prediction as the common method.
10. For the Linear Family, also investigate Bayesian Ridge intervals and
    residual bootstrap intervals.

The primary nominal coverage level will be **90%**, so `alpha = 0.10`.

### Confirmed prerequisites

| Prerequisite                  | Status | Evidence                                                     |
| ----------------------------- | ------ | ------------------------------------------------------------ |
| Corrected modeling data       | Ready  | 2,048 rows across 164 publications                           |
| Shared publication split      | Ready  | 1,434 train, 307 validation, 307 test rows                   |
| Publication separation        | Ready  | 115 train, 25 validation, 24 test publications; zero overlap |
| Predictor/metadata separation | Ready  | No publication or lineage columns in `X`                     |
| Selected model                | Ready  | Elastic Net selected by validation-publication RMSE          |
| Frozen model settings         | Ready  | Hyperparameter tuning disabled in the saved Week 8 config    |
| Bayesian Ridge settings       | Ready  | Tuned Bayesian Ridge parameters remain in the frozen config  |
| Final test baseline           | Ready  | RMSE 34.465 MPa, MAE 28.226 MPa, R2 -0.292                   |
| Thresholded LOPO set          | Ready  | Six publications with at least 50 rows; 446 held-out rows    |

## Two Important Week 9 Adaptations

### 1. Do not reuse the saved final Week 8 model for conformal calibration

The saved Week 8 Elastic Net was fitted on **train plus validation** before its
single final test evaluation. Validation therefore cannot be used to calibrate
prediction intervals around that fitted object.

For Week 9 shared-split calibration:

1. Load the frozen Week 8 configuration.
2. Rebuild the exact selected Elastic Net pipeline.
3. Fit it on Week 8 training publications only.
4. Predict the unchanged validation publications.
5. Estimate interval size from validation residuals.
6. Evaluate once on the unchanged Week 8 test publications.

The Week 8 joblib file remains the point-prediction baseline, but the Week 9
runner must save a new train-only calibrated model artifact.

### 2. Do not attach intervals to the old Week 8 LOPO predictions

The Week 8 LOPO model for each publication was trained on every other
publication. It had no independent calibration set, so its predictions cannot
receive valid split-conformal intervals after the fact.

Week 9 must rerun each of the six eligible LOPO folds with three disjoint roles:

- **held-out:** the current publication being evaluated;
- **calibration:** the fixed Week 8 validation publications, excluding the
  current held-out publication when necessary;
- **training:** every remaining publication.

This reuses the saved Week 8 publication manifest and avoids introducing a new
target-dependent split. Every fold must save and audit its role manifest.

## Fixed Scientific Questions

1. Does a nominal 90% interval actually cover about 90% of unseen-publication
   test rows?
2. Which test publications have the lowest interval coverage?
3. Which publications have wide intervals, and which have narrow but incorrect
   intervals?
4. Are the high-error Week 8 publications also poorly calibrated?
5. Do adaptive interval methods become wider for unfamiliar publications?
6. Does uncertainty quality remain similar in the shared test and six-fold
   LOPO experiments?
7. Which rows or publications should be highlighted for later explainability?

## Experiment 1: Elastic Net Split-Conformal Interval

This is the required common method and the primary Week 9 result.

Using the train-only fitted Elastic Net, calculate validation scores:

```text
score_i = abs(y_validation_i - prediction_validation_i)
```

For `n` calibration rows and `alpha = 0.10`, use the finite-sample conformal
quantile:

```text
quantile_level = ceil((n + 1) * (1 - alpha)) / n
q_hat = higher_quantile(validation_scores, quantile_level)
```

Create test intervals:

```text
lower = prediction - q_hat
upper = prediction + q_hat
```

This interval has constant width. It is a strong calibration baseline but
cannot become wider for an individually unfamiliar row. That limitation must
be stated explicitly rather than interpreted as a bug.

## Experiment 2: Bayesian Ridge Uncertainty

Use the frozen Week 8 Bayesian Ridge hyperparameters and the same train-only
preprocessing rules.

Report two Bayesian variants:

1. **Native Bayesian interval**
   - obtain predictive mean and standard deviation from Bayesian Ridge;
   - use `mean +/- 1.64485 * standard_deviation` for a nominal 90% interval;
   - evaluate whether this native interval is already calibrated.
2. **Conformalized Bayesian interval**
   - calculate validation scores
     `abs(y - mean) / max(prediction_std, epsilon)`;
   - estimate `q_hat` on validation only;
   - use `mean +/- q_hat * prediction_std` on test.

The conformalized interval remains wider where Bayesian predictive uncertainty
is larger while using validation data to correct its scale.

Bayesian Ridge is an uncertainty comparator, not a replacement chosen using
the Week 9 test set. Elastic Net remains the frozen selected Week 8 model.

## Experiment 3: Elastic Net Residual Bootstrap

Use a reproducible residual bootstrap around the frozen Elastic Net pipeline.

Recommended procedure:

1. Fit Elastic Net on training publications only.
2. Center its training residuals.
3. For each bootstrap repetition:
   - sample centered residuals with replacement;
   - create pseudo-target values from fitted values plus sampled residuals;
   - refit the complete train-only pipeline;
   - predict validation and test rows;
   - include sampled residual noise so the result represents prediction rather
     than only coefficient uncertainty.
4. Use the 5th and 95th bootstrap percentiles as the raw 90% interval.
5. Conformalize the raw bootstrap interval using validation nonconformity:

```text
score_i = max(lower_i - y_i, y_i - upper_i, 0)
```

6. Expand the test bounds by the finite-sample validation quantile.

Use a fixed seed and a configured number of repetitions. Start with 300
repetitions; increase only if interval estimates are visibly unstable.

## Experiment 4: Shared Publication-Held-Out Calibration

Run all predeclared interval methods on the existing Week 8 split:

| Role                   |  Rows | Publications | Permitted use                              |
| ---------------------- | ----: | -----------: | ------------------------------------------ |
| Training               | 1,434 |          115 | Fit preprocessing and models               |
| Validation/calibration |   307 |           25 | Estimate interval quantiles or corrections |
| Final test             |   307 |           24 | Report coverage and width once             |

The final test target must not influence method settings, interval scale,
bootstrap count, model choice, or any other decision.

Because the selected Week 8 configuration was frozen before Week 9, the
course-aligned primary analysis may use the existing validation split for
calibration. The report should still acknowledge that this validation split
was involved in Week 8 model selection. A stricter optional sensitivity check
can reserve a new calibration-publication subset from Week 8 training data.

## Experiment 5: Thresholded LOPO Calibration

Use the Week 8 threshold without redefining it in Week 9:

```text
minimum_rows_for_reliable_metrics = 50
```

The six eligible publications are:

- `Ref-144-Research`: 110 rows
- `Ref-121-Research`: 78 rows
- `Ref-48-Research`: 72 rows
- `Ref-141-Research`: 71 rows
- `Ref-85-Research`: 64 rows
- `Ref-139-Research`: 51 rows

For each publication:

1. Assign that entire publication to held-out evaluation.
2. Remove it from the fixed calibration-publication set if present.
3. Fit fresh preprocessing and model pipelines on all remaining training-role
   publications.
4. Estimate interval scale using calibration-role publications only.
5. Predict intervals for the held-out publication.
6. Save row-level intervals and publication-level calibration metrics.

Report LOPO results as:

- **micro:** every held-out row has equal weight;
- **macro:** every eligible publication has equal weight.

The shared test and LOPO metrics are complementary. They should not be merged
into one score because their training and calibration compositions differ.

## Calibration Metrics

For every method and evaluation scheme, report:

- nominal coverage;
- empirical coverage;
- coverage gap: `empirical_coverage - nominal_coverage`;
- mean interval width;
- median interval width;
- interval-width standard deviation;
- lower-side miss rate: actual value below the lower bound;
- upper-side miss rate: actual value above the upper bound;
- Winkler interval score;
- point-prediction MAE, RMSE, R2, Bias, and MedianAE;
- Spearman relationship between uncertainty width and absolute error when
  interval width varies.

For a 90% interval, the Winkler score is:

```text
width
+ 20 * (lower - actual), when actual < lower
+ 20 * (actual - upper), when actual > upper
```

Lower Winkler score is better. Interval width must never be interpreted without
coverage: a narrow interval with poor coverage is not a successful interval.

Report coverage at three levels:

1. overall row-level coverage;
2. publication-level coverage for each publication, always with `n_rows`;
3. macro coverage, where each publication receives equal weight.

Small shared-test publications may have unstable individual coverage. Keep
them in the table but flag low row counts instead of making strong conclusions.

## Overconfidence and Underconfidence Diagnostics

### Row level

Save:

- `covered`;
- lower and upper bounds;
- interval width and half-width;
- absolute error;
- uncertainty score;
- standardized error relative to interval half-width;
- miss direction.

Rows outside the interval are overconfident failures. Rank them by absolute
error and standardized error.

### Publication level

Use coverage and width together:

- **narrow but wrong:** coverage below nominal and width below the median;
- **wide but safe:** coverage at or above nominal and width above the median;
- **wide and wrong:** poor coverage despite wide intervals;
- **well calibrated:** coverage close to nominal with comparatively narrow
  intervals.

Treat these labels as diagnostics and always show publication row counts.

Merge Week 9 calibration results with Week 8 publication RMSE, Bias,
out-of-training-range rate, and unseen-category rate. This directly tests
whether Week 8 failures receive appropriately larger uncertainty.

## Required Plots

Use Matplotlib only and follow the Week 8 output conventions.

1. Overall coverage and mean width by method.
2. Empirical-versus-nominal coverage curve, preferably at 50%, 80%, 90%, and
   95% nominal levels.
3. Publication coverage ranking with a 90% reference line and row-count labels.
4. Publication mean-width ranking.
5. Coverage-versus-width diagnostic identifying narrow-but-wrong cases.
6. Prediction intervals for the worst shared-test publications.
7. Prediction intervals for the six LOPO publications.
8. Interval width or uncertainty score versus absolute error.
9. Interval width versus Week 8 shift diagnostics.
10. Native versus calibrated Bayesian and bootstrap interval comparison.

Avoid an unreadable plot containing all 307 test intervals. Use selected
publications or the largest standardized failures for detailed interval plots.

## Architecture

Follow the Week 8 package convention:

```text
configs/week09_uncertainty_calibration.yaml
src/s1_linear/week09/__init__.py
src/s1_linear/week09/data.py
src/s1_linear/week09/splits.py
src/s1_linear/week09/calibration.py
src/s1_linear/week09/methods.py
src/s1_linear/week09/experiments.py
src/s1_linear/week09/plots.py
src/s1_linear/week09/runner.py
tests/test_week09_calibration.py
WEEK09_STRATEGY.md
notebooks/week09_uncertainty_calibration.ipynb
```

Responsibilities:

- `data.py`: load and validate Week 8 splits, lineage, selected model, and
  frozen configuration.
- `splits.py`: create and audit train/calibration/held-out role manifests for
  LOPO without using target values.
- `calibration.py`: conformal quantiles, interval construction, coverage,
  width, and interval-score metrics.
- `methods.py`: train-only model fitting, Bayesian predictive standard
  deviation, and residual bootstrap generation.
- `experiments.py`: shared-test and thresholded-LOPO experiment orchestration.
- `plots.py`: all Week 9 figures using Matplotlib only.
- `runner.py`: load config, run experiments, save artifacts, print summaries,
  and fail if any leakage audit fails.

Run the final workflow with:

```bash
PYTHONPATH=src .venv/bin/python -m s1_linear.week09.runner
```

## Configuration Decisions

The Week 9 config should contain:

- paths to Week 8 split data, modeling data, lineage, manifest, frozen config,
  and Week 8 diagnostic tables;
- `alpha: 0.10`;
- `epsilon` for normalized scores;
- bootstrap repetitions and random seed;
- enabled interval methods;
- figure and table output names;
- the Week 8 config path from which the 50-row LOPO threshold is read.

Do not duplicate the LOPO threshold in two independent configs. Week 9 should
read the value from Week 8 or explicitly verify that any override matches it.

## Output Structure

```text
data/processed/week9/
reports/tables/week09/
reports/figures/week09/
results/models/week09/
results/metrics/week09/
results/predictions/week09/
```

Required tables include:

- Week 9 input-readiness audit;
- shared train/calibration/test leakage audit;
- LOPO role manifests and overlap audits;
- validation calibration quantiles;
- overall shared-test calibration metrics;
- shared-test publication calibration table;
- row-level shared-test intervals;
- LOPO micro/macro calibration summary;
- LOPO publication calibration table;
- row-level LOPO intervals;
- overconfident rows and publications;
- underconfident or unnecessarily wide publications;
- method comparison table.

## Tests and Acceptance Criteria

### Unit tests

1. Finite-sample conformal quantile uses the configured higher-quantile rule.
2. Interval bounds satisfy `lower <= prediction <= upper` where appropriate.
3. Coverage, width, miss direction, and Winkler score match hand-calculated
   examples.
4. Normalized conformal scores handle zero standard deviation using epsilon.
5. Bootstrap output is deterministic for a fixed seed.
6. Publication metadata never enters model features.
7. Training, calibration, and held-out publication sets have zero overlap.
8. Held-out targets are never used to estimate interval scale.
9. The Week 8 test target is never used before final shared-test reporting.
10. Every LOPO publication satisfies the Week 8 50-row threshold.

### Workflow acceptance criteria

- All preprocessing is fitted on training-role rows only.
- All interval corrections are estimated on calibration-role rows only.
- Model configurations are loaded from frozen Week 8 settings.
- No Week 9 model or method is selected using final-test coverage.
- Every saved prediction has lineage, point prediction, lower bound, upper
  bound, width, coverage flag, and miss direction.
- Shared-test and LOPO results are reported separately.
- Micro and macro publication coverage are both available.
- The notebook explains why narrow intervals can still be overconfident.
- The complete runner and every notebook code cell execute without error.

## Recommended Implementation Order

1. Create the Week 9 package, config, and input-readiness audit.
2. Implement and test interval metrics and finite-sample conformal quantiles.
3. Implement the shared Elastic Net split-conformal experiment.
4. Implement native and conformalized Bayesian Ridge intervals.
5. Implement raw and conformalized residual bootstrap intervals.
6. Build the shared-test publication calibration table and plots.
7. Build and audit LOPO train/calibration/held-out manifests.
8. Run the six-publication LOPO calibration experiments.
9. Create overconfidence and underconfidence diagnostics.
10. Run the full workflow, freeze outputs, and create the explanatory notebook.

## Scope Guard

Week 9 is an uncertainty-calibration task, not another model-search week.

Do not:

- retune Elastic Net using Week 9 test results;
- replace the selected model because another interval looks better on test;
- fit preprocessing on validation or test rows;
- calibrate using final-test or held-out-publication residuals;
- use publication metadata as predictors;
- reuse Week 8 train-plus-validation predictions as conformal inputs;
- compare interval width without also comparing coverage;
- introduce tree-based models into the S1 Linear workflow.

The final Week 9 conclusion should separate three ideas clearly:

1. point-prediction accuracy;
2. empirical interval calibration;
3. interval sharpness or usefulness.
