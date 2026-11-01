# Optimized logistic regression baselines

## Dataset and preprocessing

Both notebooks load these files from `processed_output/`:

- `X_processed.csv`: input features.
- `y_processed.csv`: outcome labels.
- `dataset_metadata.json`: source dataset information.

They check that rows align, features are numeric and finite, labels are binary, and the data matches the source metadata.

The revised dataset contains:

- **439,887 rows and 51 features.**
- **17,587 adverse outcomes (3.9981%).**
- No missing or infinite feature values, and no constant features.
- 832 duplicate feature vectors, which are kept because identical features do not necessarily mean duplicate patients.

`processed_dataset_analysis.json` records feature ranges, class counts, validation results, and SHA-256 hashes that identify the exact source files. Each model’s metadata includes these same hashes.

### How scaling works

Scaling is built into each model pipeline and happens after the data is split:

1. During cross-validation, each training fold fits its own `StandardScaler`. That scaler then transforms the corresponding validation fold.
2. The winning pipeline is refitted on the full training split.
3. Its fitted scaler transforms the held-out test set.

Both continuous and binary numeric features are scaled.

**Pass processed records directly to the saved pipeline. Do not scale them separately.**

### Preprocessing limitations

In the revised CSV, age, BMI, and dialysis years have means near zero and standard deviations near one, consistent with prior standardization. Other transformations or missing-value replacements may also have happened upstream.

The upstream `Healthcare1C.ipynb` still contains a cell that scales the full dataset, and the source metadata does not document the preprocessing history. The CSV values alone cannot prove that all earlier processing avoided data leakage.

Patient identifiers are unavailable, so the split cannot ensure that records from the same patient stay together. These limitations apply to both baselines.

## Saved files and loading

Each baseline produces three files in `models/`. The examples below are for Baseline A; Baseline B uses `baseline_b` instead.

| File                                            | Contents                                                                                                  |
| ----------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `logistic_regression_baseline_a_best.pkl`       | Fitted pipeline containing the scaler and logistic regression model                                       |
| `logistic_regression_baseline_a_metadata.json`  | Feature order, excluded features, source hashes, settings, scores, confusion matrix, and package versions |
| `logistic_regression_baseline_a_cv_results.csv` | Aggregate scores and timings for all 200 candidates                                                       |

The pipeline expects already-processed features; it does not convert raw patient records into model inputs.

Saved pipelines remain fitted on the training split. Therefore, the held-out test scores describe the exact saved models. After saving and reloading, predicted probabilities are compared for every test record to verify that they match.

## Interpreting and sharing the models

Good ranking scores do not necessarily mean a model detects adverse outcomes well at a particular cutoff. Read adverse-class recall and the confusion matrix alongside accuracy.

To choose a different cutoff, use validation predictions generated from training data only, then evaluate it on fresh held-out data. **Do not tune a cutoff using the test results reported here.**

Including or excluding demographics does not, by itself, establish fairness. Both models remain research baselines.

Saving files locally does not upload them to GitHub. To share them, commit and push the notebooks and model artifacts through the normal Git workflow.

Exports contain fitted parameters and aggregate summaries, not individual records or predictions. Follow the dataset’s terms when sharing derived artifacts.

## Results from the revised CSVs

| Metric                     | Baseline A | Baseline B |
| -------------------------- | ---------: | ---------: |
| Feature count              |         30 |         51 |
| CV average precision       |   0.080500 |   0.083172 |
| Test average precision     |   0.080930 |   0.085460 |
| Test ROC AUC               |   0.677869 |   0.683379 |
| Test log loss              |   0.160207 |   0.159642 |
| Test balanced accuracy     |   0.500000 |   0.500000 |
| Test adverse recall at 0.5 |   0.000000 |   0.000000 |

Both models selected `class_weight=None`.

- **Baseline A:** `C=0.0003678379771828634`
- **Baseline B:** `C=0.0003125715849688235`

At the default 0.5 cutoff, neither model predicts any adverse cases. Each has **zero true positives and 3,517 false negatives**. Their roughly 96% accuracy comes from the large majority of non-adverse outcomes.

Baseline B has slightly higher ranking scores, but neither statistical significance nor fairness has been established.

At the precision shown, scores are effectively unchanged from the earlier artifacts. If inputs differ only because of earlier affine scaling, fitting a new scaler can produce the same standardized features. This update corrects the input and scaling workflow and documents what is known about preprocessing; it does not guarantee higher scores.
