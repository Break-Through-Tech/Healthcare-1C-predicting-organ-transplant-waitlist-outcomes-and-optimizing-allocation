# Logistic Regression Model Evaluation, Confusion Matrix Analysis & Official Benchmarking 

**Organization:** Break Through Tech AI Studio / MediMate Foundation  
**Milestone:** October Model Development  
**Authors / Contributors:** Healthcare 1C Team  
**Notebook:** `logistic_regression_evaluation.ipynb`

---

## Overview

This notebook evaluates the exported Baseline A and Baseline B logistic regression models using the held-out test set.

It provides:

- Reusable model evaluation functions.
- Confusion-matrix analysis and visualization.
- ROC and precision-recall curves.
- Comparison with project performance targets.
- Clinical threshold sensitivity analysis.
- A summary of findings and next steps.

## 1. Dataset and Split

The evaluation notebook loads the processed feature matrix and target:

- 439,887 patient records.
- 51 processed source features.
- Approximately 4% adverse outcomes.
- 87,978 held-out test records.
- 3,517 adverse events in the test set.

It recreates the exact stratified 80/20 split with `random_state=42` used during baseline model development.

## 2. Loaded Models

### Baseline A

- 30 clinical predictors.
- Excludes gender, race/ethnicity, and region.
- Artifact: `models/logistic_regression_baseline_a_best.pkl`.

### Baseline B

- 51 predictors.
- Includes demographic and geographic indicators.
- Artifact: `models/logistic_regression_baseline_b_best.pkl`.

## 3. Evaluation Metrics

The reusable evaluation workflow reports:

- ROC-AUC.
- PR-AUC or average precision.
- Brier score.
- Log loss.
- Accuracy.
- Balanced accuracy.
- Precision.
- Recall.
- F1 score.
- Confusion-matrix counts.

The PR-AUC and recall metrics are especially important because the adverse class represents only about 4% of observations.

## 4. Baseline Comparison

| Metric | Project Target | Baseline A | Baseline B | Result |
|---|---:|---:|---:|---|
| ROC-AUC | >= 0.78 | 0.6779 | 0.6834 | Not met |
| PR-AUC | >= 0.65 | 0.0809 | 0.0855 | Not met |
| Brier score | <= 0.18 | approximately 0.038 | approximately 0.038 | Met |
| C-index | >= 0.72 | Not applicable | Not applicable | Survival analysis required |

### Clinical interpretation

Baseline B improves ROC-AUC by only approximately 0.005 compared with Baseline A. Demographic features therefore provide a small observed performance change, not a large advantage.

The Brier score indicates relatively good probability calibration for the current test set. This does not overcome weak ranking performance or eliminate the need for fairness and clinical validation.

## 5. Default Threshold Finding

At the default 0.5 probability cutoff:

- Both baselines report approximately 96% accuracy.
- Both baselines predict zero adverse events.
- Adverse-event recall is 0%.
- The confusion matrices contain a large number of false negatives.

Accuracy is therefore unsuitable as the primary metric. It mainly reflects correct prediction of the majority negative class.

## 6. Threshold Sensitivity

The standard threshold is not required for every clinical use case.

A lower threshold can increase adverse-event recall, but it also increases:

- False positives.
- Follow-up workload.
- Potential patient anxiety or unequal downstream burden.

A threshold must be:

- Selected from training or cross-validation data.
- Reviewed by clinical stakeholders.
- Evaluated on a fresh held-out set.
- Tested across demographic and geographic subgroups.

## 7. Summary

The logistic regression baselines establish a reproducible starting point:

- Baseline A is moderately discriminative without demographic features.
- Baseline B achieves a small incremental improvement.
- Both require substantial improvement in ranking metrics.
- Both miss all adverse events at the default 0.5 threshold.
- The probability calibration metric is comparatively strong.

## Key Takeaways

- Medical variables carry substantial predictive signal.
- Demographic features do not compensate for weak overall discrimination.
- A 96% accuracy score hides the zero adverse-event recall.
- Threshold selection is a clinical decision rather than a default software setting.
- Future work should include nonlinear models, survival analysis, calibration, and fairness analysis.

## Next Steps

- Develop tree-based or boosting models.
- Analyze nonlinear interactions and feature effects.
- Evaluate survival models for time-to-event performance.
- Choose a clinically meaningful threshold from validation data.
- Report confidence intervals and uncertainty.
- Conduct subgroup fairness audits for race, sex, region, and other relevant groups.
- Implement grouped cross-validation when patient identifiers are available.
- Compare model improvements against clinical, calibration, and fairness requirements.

## Limitations

- Logistic regression assumes linear relationships in log-odds.
- Test-set threshold selection can cause optimistic bias.
- The project cannot confirm patient-level leakage without patient identifiers.
- Fairness cannot be established by excluding demographics alone.
- ROC-AUC and average precision do not represent clinical benefit without threshold context.
