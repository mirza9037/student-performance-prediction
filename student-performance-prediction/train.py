"""Run the complete research experiment: python train.py."""
from __future__ import annotations

import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "2")

import importlib.metadata
import json
import platform
import shutil
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits

from src.config import FIGURES, MODEL_PATH, REPORTS, SEED, STAGES, STAGE_LABELS, THRESHOLD
from src.data import load_data, split_indices
from src.evaluation import bootstrap_intervals, calculate_metrics, calibration_bins, fairness_audit, save_figures
from src.modeling import compare_models, make_model
from src.reporting import write_report


def main() -> None:
    """Select on training CV only; evaluate frozen selections once on shared test rows."""
    REPORTS.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame, y, metadata = load_data()
    train_indices, test_indices = split_indices(y)
    X_train, X_test = frame.iloc[train_indices], frame.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    print(f"Official data: {metadata['rows']} rows; targets: {metadata['original_target_counts']}", flush=True)
    print(f"Training: {len(y_train)}; untouched test: {len(y_test)}", flush=True)
    metadata.update({"train_size": len(y_train), "test_size": len(y_test), "random_state": SEED,
                     "train_prevalence": float(y_train.mean()), "test_prevalence": float(y_test.mean()),
                     "trained_at_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
                     "versions": {name: importlib.metadata.version(name) for name in
                                  ["numpy", "pandas", "scikit-learn", "matplotlib", "seaborn", "plotly", "streamlit", "joblib"]}})
    bundle = {"schema_version": 1, "metadata": metadata, "threshold": THRESHOLD, "stages": {}}
    comparisons, folds = [], []
    # Complete ALL model selection before evaluating either stage on the holdout.
    for stage, features in STAGES.items():
        comparison, fold_scores, selected = compare_models(X_train, y_train, features, stage)
        comparisons.append(comparison)
        folds.append(fold_scores)
        bundle["stages"][stage] = {"model_name": selected, "features": features}
        print(f"Selected for {stage}: {selected} (training CV)", flush=True)
    comparison = pd.concat(comparisons, ignore_index=True)
    comparison.to_csv(REPORTS / "model_comparison.csv", index=False)
    pd.concat(folds, ignore_index=True).to_csv(REPORTS / "cv_fold_metrics.csv", index=False)

    tests, audits, importances, intervals, bins = [], [], [], [], []
    for stage, features in STAGES.items():
        result = bundle["stages"][stage]
        model = make_model(result["model_name"], features)
        model.fit(X_train, y_train)
        probability = model.predict_proba(X_test)[:, 1]
        raw_probability = model.calibrated_classifiers_[0].estimator.predict_proba(X_test)[:, 1]
        scores = calculate_metrics(y_test, probability)
        raw_scores = calculate_metrics(y_test, raw_probability)
        baseline = calculate_metrics(y_test, np.full(len(y_test), y_train.mean()))
        tests.append({"stage": stage, "model": result["model_name"], **scores,
                      "raw_brier_score": raw_scores["brier_score"], "raw_log_loss": raw_scores["log_loss"],
                      "baseline_brier_score": baseline["brier_score"], "baseline_pr_auc": baseline["pr_auc"]})
        audits.append(fairness_audit(y_test, probability, X_test["Gender"], stage))
        intervals.append(bootstrap_intervals(y_test, probability, stage))
        bins.append(calibration_bins(y_test, probability, stage))
        print(f"Evaluating {stage}: {scores}", flush=True)
        permutation = permutation_importance(model, X_test[features], y_test, scoring="average_precision",
                                             n_repeats=10, random_state=SEED, n_jobs=1)
        importance = pd.DataFrame({"stage": stage, "feature": features, "importance_mean": permutation.importances_mean,
                                   "importance_std": permutation.importances_std}).sort_values("importance_mean", ascending=False)
        importances.append(importance)
        save_figures(y_test, probability, raw_probability, importance, FIGURES / stage, STAGE_LABELS[stage])
        result.update({"model": model, "metrics": scores,
                       "confusion_matrix": confusion_matrix(y_test, probability >= THRESHOLD, labels=[0, 1]).tolist(),
                       "input_defaults": {feature: float(X_train[feature].median()) for feature in features},
                       "input_ranges": {feature: [float(X_train[feature].min()), float(X_train[feature].max())] for feature in features}})
    test_results = pd.DataFrame(tests)
    fairness = pd.concat(audits, ignore_index=True)
    importance = pd.concat(importances, ignore_index=True)
    test_results.to_csv(REPORTS / "test_metrics.csv", index=False)
    fairness.to_csv(REPORTS / "fairness_audit.csv", index=False)
    importance.to_csv(REPORTS / "feature_importance.csv", index=False)
    pd.concat(intervals, ignore_index=True).to_csv(REPORTS / "test_confidence_intervals.csv", index=False)
    pd.concat(bins, ignore_index=True).to_csv(REPORTS / "calibration_bins.csv", index=False)
    (REPORTS / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    # Required root figure names show first-semester results. Both stages have full sets.
    for figure in (FIGURES / "first_semester").glob("*.png"):
        shutil.copy2(figure, FIGURES / figure.name)
    write_report(metadata, comparison, test_results, fairness, importance)
    temporary = MODEL_PATH.with_suffix(".tmp")
    joblib.dump(bundle, temporary, compress=3)
    temporary.replace(MODEL_PATH)
    print(f"Complete. Model: {MODEL_PATH}\nReport: {REPORTS / 'research_report.md'}", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
