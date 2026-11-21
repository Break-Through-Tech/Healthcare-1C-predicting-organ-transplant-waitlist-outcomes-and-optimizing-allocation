# Logistic Regression Model A Training, & Hyper-parameter Tuning
**Organization:** Break Through Tech AI Studio / MediMate Foundation  
**Milestone:** October Model Development  
**Notebook:** `logistic_regression_modeling_Baseline A.ipynb`

---

## Overview

Baseline A trains and evaluates a logistic regression model using 30 clinical predictors. It excludes gender, ethnicity/race, and OPTN region while keeping the shared 80/20 stratified train/test split used across baselines.

The notebook also:

- Validates the processed feature and target files.
- Uses a leakage-safe scikit-learn pipeline.
- Searches 200 regularization and class-weight configurations.
- Evaluates the selected model on the held-out test set.
- Exports the fitted model, metadata, and cross-validation results.

## 1. Input Data

The notebook loads `processed_output/X_processed.csv`, `processed_output/y_processed.csv`, and `processed_output/dataset_metadata.json`.

Validation checks confirm:

- 439,887 aligned feature rows.
- 51 numeric source predictors.
- 17,587 adverse outcomes, or approximately 4% of observations.
- No missing or infinite feature values.
- Matching metadata, class distributions, and feature names.
- Unique row and column identifiers.

The 832 duplicate feature rows are retained because matching feature values do not prove that the records represent the same patient.

## 2. Feature Selection

Baseline A removes:

- Gender columns.
- Race/ethnicity columns.
- Region columns.

This leaves 30 clinical predictors. No patient rows are removed. Baseline A therefore measures performance when sensitive demographics are not available to the model.

## 3. Data Splitting

The dataset is split once with:

- 80% for training.
- 20% for testing.
- Stratification by the target.
- `random_state=42`.

The test set is held out for final reporting. It is not used to tune hyperparameters or choose a classification threshold.

## 4. Logistic Regression Pipeline

The model pipeline applies:

1. `StandardScaler`.
2. `LogisticRegression` with the `newton-cholesky` solver.

The scaler is placed inside the pipeline so that cross-validation fits scaling parameters only on each training fold.

The hyperparameter search evaluates:

- 100 values of `C` on a logarithmic scale from 0.0001 to 1000.
- `class_weight=None` and `class_weight="balanced"`.
- Five stratified folds.

This produces 200 configurations and 1,000 validation fits.

Average precision is used to select the model because adverse outcomes are uncommon.

## 5. Held-Out Results

The reported held-out metrics are:

| Metric | Baseline A |
|---|---:|
| ROC-AUC | 0.677869 |
| Average precision | 0.080930 |
| Accuracy | 0.960024 |
| Balanced accuracy | 0.500000 |
| Adverse recall at 0.5 | 0.000000 |
| Adverse predictions at 0.5 | 0 |
| Log loss | 0.160207 |

The 96% accuracy is misleading because nearly every patient in the test set is negative. At the default 0.5 threshold, the model predicts no adverse events.

## 6. Exported Artifacts

The notebook exports:

- `models/logistic_regression_baseline_a_best.pkl`
- `models/logistic_regression_baseline_a_metadata.json`
- `models/logistic_regression_baseline_a_cv_results.csv`
- `models/processed_dataset_analysis.json`

The pickle contains the fitted pipeline, including the scaler. Reusing this artifact requires processed numeric features in the recorded feature order. The model does not accept raw patient records or perform upstream preprocessing.

## Summary

Baseline A demonstrates a valid clinical-baseline workflow but does not meet the project's ROC-AUC or PR-AUC targets:

- Held-out ROC-AUC is approximately 0.68.
- Held-out average precision is approximately 0.081.
- The default 0.5 threshold misses all adverse events.
- Calibration and threshold selection require careful, training-only analysis.

## Key Takeaways

- Medical predictors provide meaningful ranking information.
- Logistic regression captures moderate discrimination but limited separation.
- High accuracy reflects class imbalance rather than adverse-event detection.
- Demographics should not be added solely to improve a score without a fairness analysis.
- A more flexible model or survival approach may be needed to improve performance.

## Limitations

- Patient identifiers are unavailable, so grouped cross-validation cannot be verified.
- CSV contents alone cannot prove that every upstream preprocessing step was leakage-free.
- A test-derived threshold would bias held-out evaluation.
- Subgroup fairness has not been established.
- Model artifacts should be loaded only from trusted sources with compatible dependency versions.

