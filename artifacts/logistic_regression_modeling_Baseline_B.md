# Baseline B Logistic Regression Modeling & Hyper-parameter Tuning

**Organization:** Break Through Tech AI Studio / MediMate Foundation  
**Milestone:** October Model Development  
**Notebook:** `03_logistic_regression_modeling_Baseline B.ipynb`

---

## Overview

Baseline B trains and evaluates a logistic regression model using all 51 processed predictors, including gender, race/ethnicity, and region indicators. It otherwise follows the same workflow as Baseline A.

The notebook validates the source data, constructs a leakage-safe pipeline, performs hyperparameter search, exports the fitted model, and reports held-out test metrics.

## 1. Input Data

The notebook loads:

- `processed_output/X_processed.csv`
- `processed_output/y_processed.csv`
- `processed_output/dataset_metadata.json`

The data contains 439,887 rows, 51 numeric predictors, and 17,587 adverse outcomes. The target prevalence is approximately 4%, creating substantial class imbalance.

The notebook validates row alignment, target values, metadata consistency, feature uniqueness, missing values, infinite values, and source SHA-256 hashes.

## 2. Feature Selection

Baseline B retains all 51 processed predictors, including:

- Clinical and biologic variables.
- One-hot encoded gender.
- One-hot encoded race/ethnicity.
- One-hot encoded geographic region.

Demographic inclusion is intentional for comparative fairness and performance analysis. It does not, by itself, establish that a model is fair or appropriate for allocation decisions.

## 3. Data Splitting

The model uses an 80/20 stratified split with `random_state=42`. The same split is used for Baseline A and Baseline B.

The test set is reserved for final evaluation and is not used for hyperparameter or threshold selection.

## 4. Logistic Regression Pipeline

The pipeline contains:

1. `StandardScaler`.
2. `LogisticRegression` with the `newton-cholesky` solver.

The scaler is fit only within training folds during cross-validation and on the complete training split before test-set transformation.

The search includes:

- 100 logarithmic values of `C`.
- `class_weight=None` and `class_weight="balanced"`.
- Five stratified folds.
- 200 configurations and 1,000 validation fits.

The winning configuration is selected using mean cross-validation average precision.

## 5. Held-Out Results

The reported held-out metrics are:

| Metric | Baseline B |
|---|---:|
| ROC-AUC | 0.683379 |
| Average precision | 0.085460 |
| Accuracy | 0.960024 |
| Balanced accuracy | 0.500000 |
| Adverse recall at 0.5 | 0.000000 |
| Adverse predictions at 0.5 | 0 |
| Log loss | 0.160177 |

Adding demographics produces a small improvement relative to Baseline A but does not yield acceptable adverse-event recall at the default threshold.

## 6. Exported Artifacts

The notebook exports:

- `models/logistic_regression_baseline_b_best.pkl`
- `models/logistic_regression_baseline_b_metadata.json`
- `models/logistic_regression_baseline_b_cv_results.csv`
- `models/processed_dataset_analysis.json`

The exported pickle is reload-verified. It expects the complete processed feature matrix in metadata feature order without a second scaler.

## Summary

Baseline B is a valid all-feature logistic regression baseline:

- Held-out ROC-AUC is approximately 0.683.
- Held-out average precision is approximately 0.085.
- Performance exceeds Baseline A slightly.
- The default 0.5 cutoff predicts no adverse events.
- Clinical ranking remains limited with a linear model.

## Key Takeaways

- Demographic predictors add a small observed performance difference.
- Clinical information remains likely to contribute most of the ranking signal.
- Accuracy is not an appropriate primary success metric in this imbalanced problem.
- Threshold analysis must prioritize adverse-event recall or another clinically justified trade-off.
- Fairness testing is required before demographic features are used in a consequential system.

## Limitations

- The available data does not identify individual patients, preventing patient-grouped validation.
- The source metadata does not fully prove leakage-free upstream preprocessing.
- No subgroup fairness metrics are reported.
- The test set is appropriate for final reporting, not threshold optimization.
- Logistic regression cannot model nonlinear interactions as flexibly as tree or boosting methods.

