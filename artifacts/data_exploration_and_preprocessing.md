# Data Exploration and Preprocessing

**Organization:** Break Through Tech AI Studio / MediMate Foundation  
**Milestone:** November Model Development  
**Notebook:** `data exploration_and_preprocessing.ipynb`

---

## Overview

This notebook converts raw kidney transplant waitlist data into a processed feature matrix for the downstream baseline models.

The primary objectives are to:

- Define a one-year adverse outcome.
- Engineer features available at or before initial waitlist listing.
- Prevent use of future or competing-outcome information.
- Address missing values, outliers, skewness, categorical data, and feature scale.
- Export a consistent numeric matrix and target vector.

## 1. Target Definition

The modeling horizon is 365 days after initial listing.

The adverse class (`1`) includes candidates who:

- Died within 365 days, or
- Were removed from the waitlist as too sick within 365 days.

The negative class (`0`) includes candidates who:

- Remained adverse-event-free for at least 365 days, or
- Received a transplant before 365 days.

Invalid follow-up dates and unknown outcomes are removed. Zero-day outcomes are retained when their listing and event dates are internally consistent.

The final modeling subset contains 439,887 records:

- 422,300 negative outcomes.
- 17,587 adverse outcomes.
- Approximately 96% negative and 4% positive.

## 2. Leakage Prevention

Several variables are handled to avoid future information leakage:

- `END_CPRA` is excluded because it reflects later measurements.
- Dialysis duration is calculated only from dialysis beginning at or before listing.
- Patients who began dialysis after listing receive a zero duration.
- Post-listing outcomes are not used as predictors.

Multiple waitlist registrations occur for many patients. Patient identifiers should be retained for future grouped train/test and cross-validation splits.

## 3. Clinical Feature Engineering

Important engineered predictors include:

- `dialysis_at_listing`: whether dialysis had started by listing.
- `dialysis_years_at_listing`: pre-listing dialysis duration.
- `A2A2B_STATUS`: eligibility and applicability categories.
- `INIT_CPRA_CLEAN`: initial calculated panel reactive antibody.
- `FUNC_STATUS_PERCENT`: parsed functional-status percentage.
- `FUNC_STATUS_SCALE`: adult, pediatric, unknown, or not-applicable scale category.
- `BMI_TCR_CLEAN`: cleaned body mass index.

The BMI cleaning step limits implausible values to a biologically plausible range before imputation.

## 4. Missing Data

A missingness indicator is created for calculated panel reactive antibody where appropriate.

Median imputation is applied to:

- `BMI_TCR_CLEAN`
- `INIT_CPRA_CLEAN`
- `FUNC_STATUS_PERCENT`

Missing categorical values are represented with an explicit `unknown` category.

Missingness also varies by gender, race/ethnicity, and geographic region. These patterns may reflect data quality or administrative disparities rather than biological differences.

## 5. Outliers and Distribution Treatment

Continuous variables are treated with:

- IQR winsorization for extreme outliers.
- Negative-log transformation for left-skewed age.
- Positive `log1p` transformation for dialysis duration.
- Clinical binning of cPRA at 0, 1-20, 21-80, and 81-100.
- Density-based or predefined binning for functional status.

The goal is to reduce sensitivity to extreme values and make nonlinear clinical relationships more explicit for linear models.

## 6. Categorical Encoding

Categorical variables are imputed and encoded so they can be consumed by scikit-learn.

The one-hot encoded feature set includes:

- ABO blood type.
- Initial transplant status.
- A2/A2B status.
- Functional-status scale.
- Gender.
- Race/ethnicity.
- Geographic region.

The final matrix contains 51 features.

## 7. Feature Scaling

Non-binary numeric features are standardized with `StandardScaler`.

For downstream modeling, the notebook exports the processed features and target. Baseline notebooks also refit scaling inside the model pipeline to preserve training-fold isolation during cross-validation.

## 8. Final Data Export

The notebook exports the processed data and metadata for reuse:

- `processed_output/X_processed.csv`
- `processed_output/y_processed.csv`
- `processed_output/dataset_metadata.json`

The exported matrix has:

- 439,887 rows.
- 51 numeric features.
- Zero remaining missing values.
- A binary `0/1` target.
- A companion metadata file describing dimensions, feature names, and class counts.

## Summary

The preprocessing pipeline produces a complete, numeric, machine-learning-ready dataset for logistic regression and other models. It combines clinical feature engineering, missing-data treatment, outlier handling, categorical encoding, and scaling.

## Key Takeaways

- The adverse-event target is clearly defined and strongly imbalanced.
- Clinical variables are designed to reflect information available at listing.
- Missingness indicators preserve potentially informative absent measurements.
- Winsorization and transformations reduce raw-distribution extremes.
- The final 51-feature matrix supports both baseline A and baseline B comparisons.

## Limitations

- A complete patient identifier is not retained in the exported modeling matrix.
- Patient-level grouping is not guaranteed in downstream row-based splits.
- Median imputation and winsorization parameters require training-only fitting in future pipelines.
- Full-data scaling or transformation can appear to simplify preprocessing even when upstream provenance is incomplete.
- Missingness and demographic categories can reflect structural data-quality biases.
- Data alone does not establish that resulting predictions are fair or appropriate for organ-allocation decisions.

