# Optimized logistic regression baselines

## Dataset and preprocessing

Both notebooks load `X_processed.csv`, `y_processed.csv`, and `dataset_metadata.json` from `processed datasets/`. They validate row alignment, numeric/finite features, binary labels, and agreement with source metadata. The revised inputs contain 439,887 rows and 51 features, with 17,587 adverse outcomes (3.9981%). There are no missing/infinite feature values or constant features. The 832 duplicate feature vectors are retained: matching features do not prove duplicate patients.

`processed_dataset_analysis.json` records aggregate ranges, class counts, validation findings, and SHA-256 source-file hashes. Each model metadata file includes the same hashes so its input revision can be identified.

Scaling happens inside each pipeline, after splitting. Each CV training fold fits its own StandardScaler, which then transforms that fold's validation inputs. The winning pipeline is refitted on the full training split, and that training scaler transforms the held-out test set. Both continuous and binary numeric predictors are scaled. **Do not scale records separately before calling the saved pipeline.**

The revised CSV values are not standardized; BMI, for example, remains on its original scale. Other transformations and imputations may already have occurred upstream. The upstream `Healthcare1C.ipynb` still contains a full-data scaling cell, while the source metadata does not record preprocessing provenance. CSV values alone cannot establish that all upstream processing was leakage-free. Patient identifiers are unavailable for patient-group separation. These limitations apply to both baselines.

## Reproduction and search

Install the tested dependencies with `python -m pip install -r models/requirements.txt`, then run the corresponding notebook from top to bottom:

- `notebooks/logistic_regression_training_Baseline A.ipynb`: excludes gender, ethnicity, and region; 30 features.
- `notebooks/logistic_regression_training_Baseline B.ipynb`: retains demographics; all 51 features.

Both use the same stratified 80/20 split, seed 42, and five stratified CV folds. Each GridSearchCV tests 200 configurations: 100 L2 regularization strengths from 0.0001 to 1000 and two class-weight choices (`None`, `balanced`). The solver is `newton-cholesky`. Average precision selects the winner; ROC AUC is also recorded. Four workers are used with one numerical-library thread per worker to avoid thread oversubscription. The test set never selects hyperparameters or thresholds.

## Artifacts and loading

For each baseline, `baseline_a` or `baseline_b` appears in these filenames:

- `logistic_regression_baseline_a_best.pkl`: fitted scaler and logistic regression pipeline.
- `logistic_regression_baseline_a_metadata.json`: input feature order, exclusions, source hashes, settings, scores, confusion matrix, and package versions.
- `logistic_regression_baseline_a_cv_results.csv`: aggregate scores and timings for all 200 candidates.

Use the Python and package versions in the metadata. Load only trusted pickle files.

```python
import json
import pickle
from pathlib import Path

model_dir = Path("models")  # run from the repository root
baseline = "a"  # change to "b" to load Baseline B
prefix = f"logistic_regression_baseline_{baseline}"
metadata = json.loads((model_dir / f"{prefix}_metadata.json").read_text())
with (model_dir / f"{prefix}_best.pkl").open("rb") as handle:
    model = pickle.load(handle)

# processed_rows contains features from the revised processing workflow.
# Select the recorded columns and pass them directly, without separate scaling.
features = processed_rows.loc[:, metadata["feature_columns"]]
positive_index = list(model.classes_).index(metadata["positive_class"])
adverse_probabilities = model.predict_proba(features)[:, positive_index]
```

The model does not transform raw patient records. Exported pipelines remain fitted on the training split, so held-out scores describe those exact artifacts. Export/reload verification compares probabilities on every test record.

## Interpretation and sharing

Ranking metrics do not establish useful detection at a fixed threshold. Read adverse-class recall and the confusion matrix alongside accuracy. Select any alternative cutoff using training-only validation predictions, then assess it on fresh held-out data. Do not tune a threshold against these test results. Demographic inclusion or exclusion alone does not establish fairness. These models remain research baselines.

Files are saved locally; saving does not upload to GitHub. Commit and push the notebooks and model artifacts through the normal Git workflow to share them. Only fitted parameters and aggregate summaries are exported, not row-level records or predictions. Follow the dataset's terms for sharing derived artifacts.

## Results from the revised CSVs

| Metric | Baseline A | Baseline B |
| --- | ---: | ---: |
| Feature count | 30 | 51 |
| CV average precision | 0.080500 | 0.083172 |
| Test average precision | 0.080930 | 0.085460 |
| Test ROC AUC | 0.677869 | 0.683379 |
| Test log loss | 0.160207 | 0.159642 |
| Test balanced accuracy | 0.500000 | 0.500000 |
| Test adverse recall at 0.5 | 0.000000 | 0.000000 |

Both selected `class_weight=None`; A selected `C=0.0003678379771828634` and B selected `C=0.0003125715849688235`. Both predict no adverse cases at 0.5 (zero true positives and 3,517 false negatives). Their 96% accuracy reflects the imbalanced target. B has slightly higher ranking scores, but statistical significance and fairness have not been established. Scores are effectively unchanged at this precision from the earlier artifacts; fitting a new scaler to inputs that differ only by prior affine scaling can produce the same standardized features. The correction is to the input/scaling workflow and its documented provenance, not a guaranteed score increase.
