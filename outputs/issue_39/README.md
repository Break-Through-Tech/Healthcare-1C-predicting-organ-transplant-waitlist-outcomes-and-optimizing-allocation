# Issue 39 — Random Forest probability and cutoff evaluation

Completed November 17, 2026. Source: PR #48 head e51848eea9735cdd36ae8885ad089858e67ffb0b.

## Results and recommendation

For the prespecified exploratory FN:FP cost ratio of 10:1, use **0.065 (6.5%) only as the research cost-minimizing cutoff** for this sigmoid-calibrated development refit. It was selected on validation data using a 0.001 grid, not on test outcomes. It is not a validated clinical cutoff and should not be adopted as a broad screening policy: test sensitivity is only 22.52% (792 detected, 2,725 missed), with 5,419 false positives and 12.75% precision. The cutoff flags 7.06% of records. Test weighted cost is 32,669 versus 34,997 at the calibrated model's 0.50 cutoff.

There is no single defensible operational cutoff without an agreed cost ratio, minimum recall, and review capacity. The cost sensitivity table shows validation recall increases from 21.7% at 10:1 to 90.6% at 50:1, while the flagged fraction increases from 7.0% to 70.2%. Those tradeoffs need stakeholder review before operational use.

Raw RF mean predicted risk was 3.988%, versus observed prevalence of 3.998%. This supports calibration in the aggregate, but does not establish individual reliability. Sigmoid calibration did not improve held-out proper scoring metrics: Brier increased from 0.037272 to 0.037381 and log loss from 0.156973 to 0.158237. Do not claim recalibration improved probability realism. Both are slightly better than the prevalence-only Brier baseline of approximately 0.03838. The quantile calibration figure provides a binned view; no confidence intervals were calculated.

The highest-risk test decile contains 28.89% of all observed events, with 2.89x event-rate lift over overall prevalence.

## Reproduction

From a checkout containing PR #48 model metadata and processed data:

```bash
python scripts/issue39_random_forest_evaluation.py --repo-root . --fn-cost 10 --fp-cost 1 --calibration sigmoid
```

Executed successfully with Python's installed scikit-learn 1.8.0. The source model metadata lists 1.9.1. To avoid pickle compatibility and executable deserialization, the script reconstructs the RF from the recorded optimized hyperparameters and refits it on development data. It does not evaluate the original fitted pickle. Hyperparameter selection used the upstream training partition; validation here is therefore exploratory rather than a fully nested model-selection estimate.

Original stratified split: 351,909 training records and 87,978 test records. Development split: 80% of training; validation: 20% of training, random seed 43. Sigmoid CalibratedClassifierCV uses 3 folds on development only. Test records are excluded from fitting, calibration, and cutoff selection. Windows line-ending hashes exactly match model metadata after line-ending conversion alone.

## Files

- calibration_curve.png: held-out raw and sigmoid calibration curves.
- threshold_cost_curve.png: validation weighted cost and operating tradeoffs.
- decile_lift.png and test_decile_lift.csv: held-out enrichment by descending risk decile.
- validation_threshold_sweep.csv: grid search with confusion counts and costs.
- cost_sensitivity.csv: validation-selected thresholds for six hypothetical cost ratios.
- results_summary.json and test_selected_vs_default.csv: final evaluation numbers.

Known limits: upstream preprocessing leakage and repeat-patient leakage were not audited; the test set has already been examined in upstream work; no external validation, uncertainty intervals, or clinical cost elicitation. A 0.50 cutoff has different results for this calibrated refit than for the upstream original uncalibrated model. Do not transfer these thresholds to another fit without validation.

Verification passed: known zero-cost threshold fixture, decile event conservation, and rejection of nonfinite probabilities. Full pipeline generated all three figures and result tables. Results are submitted for review with issue #39.
