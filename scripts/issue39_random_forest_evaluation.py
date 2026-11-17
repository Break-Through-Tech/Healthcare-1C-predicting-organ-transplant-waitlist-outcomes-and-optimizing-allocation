"""Issue #39: out-of-sample RF calibration, warning cutoff and decile lift.

Run from the repository root (or with --repo-root PATH). Uses a *development*
validation split from the original training partition for cutoff selection;
the 20% held-out test set is never used to choose a threshold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, log_loss, precision_score,
                             recall_score, f1_score)
from sklearn.model_selection import train_test_split


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_probabilities(y, probability):
    y = np.asarray(y, dtype=int)
    p = np.asarray(probability, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or len(y) == 0:
        raise ValueError("Labels and probabilities must be nonempty aligned 1D arrays")
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("Expected labels with both 0 and 1")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Predicted probabilities must be finite and in [0, 1]")
    return y, p


def threshold_sweep(y, probability, fn_cost, fp_cost, grid=None):
    """Costs are hypothetical ratios, not medically validated costs."""
    y, probability = validate_probabilities(y, probability)
    if fn_cost <= 0 or fp_cost <= 0:
        raise ValueError("Cost weights must be positive")
    if grid is None:
        grid = np.unique(np.r_[0.0, np.arange(0.001, 0.501, 0.001), 1.0])
    rows = []
    for threshold in grid:
        pred = probability >= threshold
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        rows.append(dict(threshold=float(threshold), tp=int(tp), fp=int(fp),
                         fn=int(fn), tn=int(tn), flagged=int(pred.sum()),
                         flagged_fraction=float(pred.mean()),
                         recall=float(tp / (tp + fn)),
                         precision=float(tp / (tp + fp)) if tp + fp else 0.0,
                         f1=float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
                         weighted_cost=float(fn_cost * fn + fp_cost * fp),
                         mean_weighted_cost=float((fn_cost * fn + fp_cost * fp) / len(y))))
    return pd.DataFrame(rows)


def best_threshold(sweep):
    """Tie-break equal cost by fewer flags, then higher cutoff."""
    return sweep.sort_values(["weighted_cost", "flagged", "threshold"],
                             ascending=[True, True, False]).iloc[0]


def decile_lift(y, probability):
    y, probability = validate_probabilities(y, probability)
    rank = np.argsort(-probability, kind="stable")
    group = np.array_split(rank, 10)
    base_rate = float(y.mean())
    total_events = int(y.sum())
    rows = []
    captured = 0
    for i, indices in enumerate(group, start=1):
        events = int(y[indices].sum())
        captured += events
        event_rate = float(y[indices].mean())
        rows.append(dict(decile=i, records=len(indices), events=events,
                         observed_rate=event_rate,
                         lift=event_rate / base_rate,
                         cumulative_event_capture=captured / total_events,
                         mean_prediction=float(probability[indices].mean())))
    return pd.DataFrame(rows)


def metrics(y, p):
    return dict(brier=float(brier_score_loss(y, p)),
                log_loss=float(log_loss(y, p, labels=[0, 1])),
                average_precision=float(average_precision_score(y, p)),
                mean_prediction=float(np.mean(p)), observed_prevalence=float(np.mean(y)))


def plot_calibration(y, scores, output):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    for name, p in scores.items():
        observed, predicted = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(predicted, observed, "o-", label=name)
    ax.set(xlabel="Average predicted adverse-event probability",
           ylabel="Observed event frequency", title="RF calibration (held-out test)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_sweep(sweep, selected, output):
    fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(sweep.threshold, sweep.mean_weighted_cost)
    axes[0].set(ylabel="Weighted error cost per record", title="Validation cutoff sweep")
    for name in ("recall", "precision", "flagged_fraction"):
        axes[1].plot(sweep.threshold, sweep[name], label=name)
    for ax in axes:
        ax.axvline(selected, linestyle="--", color="black", label="Selected cutoff")
        ax.grid(alpha=0.25)
    axes[1].legend()
    axes[1].set(xlabel="Warning probability threshold", ylabel="Fraction")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_lift(lift, output):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(lift.decile, lift.lift)
    ax.axhline(1, color="black", linestyle="--", label="Random selection")
    ax.set(xlabel="Test-set risk decile (1 = highest predicted risk)",
           ylabel="Event-rate lift over overall prevalence", title="RF test decile lift")
    ax.set_xticks(range(1, 11))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def run(repo_root: Path, fn_cost: float, fp_cost: float, calibration: str,
        validation_fraction: float, model_name: str):
    data = repo_root / "processed_output"
    models = repo_root / "models"
    output = repo_root / "outputs" / "issue_39"
    model_path = models / f"{model_name}_best.pkl"
    metadata_path = models / f"{model_name}_metadata.json"
    for path in [metadata_path, data / "X_processed.csv", data / "y_processed.csv"]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required file: {path}. Get optimized model metadata and processed data before running.")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("target") != "adverse_365" or metadata.get("positive_class") != 1:
        raise ValueError("Unexpected target definition or positive class")
    for name in ("X_processed.csv", "y_processed.csv"):
        expected = metadata.get("source_sha256", {}).get(name)
        # Git normalizes CSV line endings; accept the original Windows hash
        # only when changing line endings alone exactly reproduces it.
        windows_hash = hashlib.sha256((data / name).read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")).hexdigest()
        if not expected or expected not in (file_hash(data / name), windows_hash):
            raise ValueError(f"Input hash differs from the model's training data: {name}")
    X = pd.read_csv(data / "X_processed.csv", index_col=0)
    y_frame = pd.read_csv(data / "y_processed.csv", index_col=0)
    if y_frame.shape[1] != 1 or not X.index.equals(y_frame.index) or not X.index.is_unique:
        raise ValueError("Features and labels are not uniquely aligned")
    y = y_frame.iloc[:, 0]
    if not np.isfinite(X.to_numpy()).all() or not set(y.unique()) == {0, 1}:
        raise ValueError("Inputs contain invalid values")
    features = metadata["feature_columns"]
    if not set(features).issubset(X.columns):
        raise ValueError("Model training features missing from data")
    X = X.loc[:, features]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=metadata["test_size"], random_state=metadata["random_state"], stratify=y)
    if len(y_train) != metadata["training_rows"] or len(y_test) != metadata["test_rows"]:
        raise ValueError("Split size does not match model metadata")
    X_dev, X_val, y_dev, y_val = train_test_split(
        X_train, y_train, test_size=validation_fraction,
        random_state=metadata["random_state"] + 1, stratify=y_train)
    # Refit chosen hyperparameters on development-only data so the validation
    # and test scores use a consistent fit. Neither validation nor test enters fitting.
    saved_model = RandomForestClassifier(**metadata["best_parameters"], random_state=metadata["random_state"], n_jobs=4)
    dev_model = clone(saved_model)
    dev_model.fit(X_dev, y_dev)
    positive_index = list(dev_model.classes_).index(1)
    val_raw = dev_model.predict_proba(X_val)[:, positive_index]
    validate_probabilities(y_val, val_raw)
    test_raw = dev_model.predict_proba(X_test)[:, positive_index]
    validate_probabilities(y_test, test_raw)

    if calibration == "none":
        val_probability, test_probability = val_raw, test_raw
        score_name = "Uncalibrated Random Forest"
    else:
        # Calibrate a fresh clone on development records via internal CV, then
        # generate validation/test predictions without including either in fit.
        calibrated = CalibratedClassifierCV(estimator=clone(saved_model), method=calibration, cv=3)
        calibrated.fit(X_dev, y_dev)
        val_probability = calibrated.predict_proba(X_val)[:, list(calibrated.classes_).index(1)]
        test_probability = calibrated.predict_proba(X_test)[:, list(calibrated.classes_).index(1)]
        score_name = f"{calibration.title()} calibrated RF"
    validate_probabilities(y_val, val_probability)
    validate_probabilities(y_test, test_probability)

    output.mkdir(parents=True, exist_ok=True)
    sweep = threshold_sweep(y_val, val_probability, fn_cost, fp_cost)
    selected = float(best_threshold(sweep).threshold)
    sweep.to_csv(output / "validation_threshold_sweep.csv", index=False)
    sensitivity = []
    for ratio in [2, 5, 10, 20, 50, 100]:
        scenario = sweep.assign(weighted_cost=ratio * sweep.fn + sweep.fp)
        winner = best_threshold(scenario)
        sensitivity.append(dict(fn_cost=ratio, fp_cost=1,
                                threshold=float(winner.threshold),
                                validation_recall=float(winner.recall),
                                validation_precision=float(winner.precision),
                                flagged_fraction=float(winner.flagged_fraction),
                                weighted_cost=float(winner.weighted_cost)))
    pd.DataFrame(sensitivity).to_csv(output / "cost_sensitivity.csv", index=False)
    lift = decile_lift(y_test, test_probability)
    lift.to_csv(output / "test_decile_lift.csv", index=False)
    plot_calibration(y_test, {"Raw RF": test_raw, score_name: test_probability}, output / "calibration_curve.png")
    plot_sweep(sweep, selected, output / "threshold_cost_curve.png")
    plot_lift(lift, output / "decile_lift.png")
    test_sweep = threshold_sweep(y_test, test_probability, fn_cost, fp_cost,
                                 grid=np.array([selected, 0.5]))
    test_sweep.to_csv(output / "test_selected_vs_default.csv", index=False)
    report = {
        "model": model_name, "target": metadata["target"],
        "calibration_method": calibration,
        "selection_split": "development validation only",
        "evaluation_fit": "refitted optimized hyperparameters on development-only split",
        "validation_fraction_of_training": validation_fraction,
        "test_rows": len(y_test), "test_adverse_events": int(y_test.sum()),
        "false_negative_cost_weight": fn_cost, "false_positive_cost_weight": fp_cost,
        "selected_threshold": selected,
        "validation_selected": best_threshold(sweep).to_dict(),
        "test_default_vs_selected": test_sweep.to_dict(orient="records"),
        "test_raw_probability_metrics": metrics(y_test, test_raw),
        "test_evaluated_probability_metrics": metrics(y_test, test_probability),
        "test_top_decile_lift": float(lift.iloc[0].lift),
        "test_top_decile_event_capture": float(lift.iloc[0].cumulative_event_capture),
        "notes": "Research-only hypothetical cost weights. Neither calibration nor thresholds are clinically validated. Upstream preprocessing and repeat-patient leakage remain unverified."
    }
    (output / "results_summary.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(json.dumps(report, indent=2, default=float))
    print("Outputs:", output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--fn-cost", type=float, default=10, help="Hypothetical false-negative cost ratio")
    parser.add_argument("--fp-cost", type=float, default=1)
    parser.add_argument("--calibration", choices=["none", "sigmoid", "isotonic"], default="sigmoid")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--model", default="random_forest_baseline_b")
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be strictly between 0 and 1")
    run(args.repo_root.resolve(), args.fn_cost, args.fp_cost,
        args.calibration, args.validation_fraction, args.model)


if __name__ == "__main__":
    main()
