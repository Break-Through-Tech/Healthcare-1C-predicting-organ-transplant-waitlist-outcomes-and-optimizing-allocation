# Optimized logistic regression ? Baseline A

Run `notebooks/logistic_regression_training_Baseline A.ipynb` from top to bottom to reproduce these artifacts. The search evaluates 200 settings using five-fold stratified cross-validation on 80% of the processed dataset. The highest mean average precision selects the model. The remaining 20% is used only for final evaluation.

- `logistic_regression_baseline_a_best.pkl`: fitted StandardScaler + LogisticRegression pipeline.
- `logistic_regression_baseline_a_metadata.json`: ordered input features, selected settings, aggregate evaluation metrics, and package versions.
- `logistic_regression_baseline_a_cv_results.csv`: scores and timings for all candidates.

The model expects already-processed numeric features from the project's existing preprocessing workflow. It does not accept raw patient records. The upstream `Healthcare1C.ipynb` fits scaling before exporting the full dataset, so these validation estimates inherit that limitation. Patient identifiers are absent from the processed inputs, so this notebook also cannot enforce patient-group separation. The exported pipeline is fitted on the training split; it is the exact pipeline measured on the held-out test set.

## Loading

Use the Python and package versions recorded in the metadata. Install the tested dependencies with `python -m pip install -r models/requirements.txt`. Load pickle files only from a trusted source.

```python
import json
import pickle
from pathlib import Path

model_dir = Path("models")  # run from the repository root
metadata = json.loads((model_dir / "logistic_regression_baseline_a_metadata.json").read_text())
with (model_dir / "logistic_regression_baseline_a_best.pkl").open("rb") as handle:
    model = pickle.load(handle)

# processed_rows is your DataFrame of already-processed records.
features = processed_rows.loc[:, metadata["feature_columns"]]
positive_index = list(model.classes_).index(metadata["positive_class"])
adverse_probabilities = model.predict_proba(features)[:, positive_index]
```

## Sharing through GitHub

A model file is a repository artifact, just like a notebook. Save it locally, then commit and push it together with this notebook, metadata, and CV results through your normal Git workflow. In a hosted notebook, download those files before adding them to your local clone. Saving alone does not upload anything to GitHub.

These exports contain fitted parameters and aggregate summaries, not patient rows. Follow the dataset's terms for sharing derived models. This is a research baseline; these results do not establish clinical validity or fairness.

## Executed results

The selected model uses L2 regularization, `C=0.0003678379771828634`, and `class_weight=None`. Mean cross-validation average precision is 0.080500. Held-out average precision is 0.080930 and ROC AUC is 0.677869.

At the default 0.5 threshold the model predicts only class 0: adverse-class recall is zero. Accuracy (0.960024) is therefore not evidence of useful adverse-case detection. The selection metric measures probability ranking across thresholds. Choose any classification threshold using training-only validation predictions, then evaluate it on fresh held-out data. Do not tune the threshold against this notebook's test set.


## Baseline B

Run `notebooks/logistic_regression_training_Baseline B.ipynb` to reproduce B. It retains all 51 processed features, including gender, ethnicity, and region; A's saved artifact uses 30 features. B evaluates 200 combinations of L2 regularization strength and class weighting with the `newton-cholesky` solver and five-fold stratified cross-validation. This matches the search that produced the saved A artifact. It uses the same stratified 80/20 split and random seed.

B exports separate files without replacing A:

- `logistic_regression_baseline_b_best.pkl`: fitted scaler and logistic regression pipeline.
- `logistic_regression_baseline_b_metadata.json`: feature order, fixed and selected settings, metrics, and software versions.
- `logistic_regression_baseline_b_cv_results.csv`: aggregate results for all 200 candidates.

Use the same `models/requirements.txt`. To load B, substitute `baseline_b` for `baseline_a` in the loading example above. Supply all features in B's metadata, including demographics; A's 30-feature inputs are insufficient for B.

The upstream preprocessing and patient-group limitations described above apply to both baselines. Demographic inclusion requires subgroup evaluation before drawing fairness conclusions. Select any alternative decision threshold using training-only validation predictions, not these test results.

### Baseline B results and comparison

| Metric | Saved Baseline A | Saved Baseline B |
| --- | ---: | ---: |
| Feature count | 30 | 51 |
| Mean CV average precision | 0.080500 | 0.083172 |
| Test average precision | 0.080930 | 0.085460 |
| Test ROC AUC | 0.677869 | 0.683379 |
| Test log loss | 0.160207 | 0.159642 |
| Test adverse recall at 0.5 | 0 | 0 |

B selected `C=0.00031257158496882353` and `class_weight=None`. Both saved pipelines predict only class 0 at the default threshold; their 96% accuracy therefore does not demonstrate adverse-case detection. B's ranking scores are slightly higher, but this comparison does not establish a statistically significant improvement or fairness. Both inherit the upstream validation limitations described above. The export/reload check confirmed matching probabilities over all 87,978 test records.
