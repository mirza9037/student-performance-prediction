"""Descriptive statistics, conditional frequencies, figures and research report."""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from src.config import SCHEMAS, TITLE
from src.modeling import bin_label


def statistical_analysis(frame: pd.DataFrame, y: pd.Series, schema: str) -> dict:
    """Describe observed data without filling gaps or implying causal effects."""
    features = SCHEMAS[schema]["features"]
    observed = frame[features].copy()
    observed["pass"] = y
    summaries = frame[features].agg(["count", "mean", "std", "min", "median", "max"]).T
    summaries.index.name = "feature"
    by_outcome = observed.groupby("pass")[features].agg(["mean", "std", "count"])
    by_outcome.index = by_outcome.index.map({0: "Fail", 1: "Pass"})
    counts = []
    for feature, edges in zip(features, SCHEMAS[schema]["bins"]):
        values = frame[feature]
        categories = np.digitize(values.fillna(0), edges)
        for category in range(4):
            mask = values.notna() & (categories == category)
            n = int(mask.sum())
            counts.append({"feature": feature, "interval": bin_label(edges, category),
                           "students": n, "passes": int(y[mask].sum()),
                           "observed_pass_rate": float(y[mask].mean()) if n else np.nan})
    return {"summary": summaries, "by_outcome": by_outcome,
            "pearson": observed.corr(method="pearson"),
            "spearman": observed.corr(method="spearman"),
            "conditional": pd.DataFrame(counts)}


def write_reports(bundle: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    for name, key in [("model_comparison", "comparison"), ("cv_fold_metrics", "fold_metrics"),
                      ("test_metrics", "test_metrics")]:
        bundle[key].to_csv(output / f"{name}.csv", index=False)
    for name, frame in bundle["statistics"].items():
        frame.to_csv(output / f"{name}.csv", index=name != "conditional")
    metadata = {**bundle["metadata"], "selected_model": bundle["model_name"],
                "train_indices": bundle["train_indices"].tolist(),
                "test_indices": bundle["test_indices"].tolist(),
                "features": SCHEMAS[bundle["schema"]]["features"],
                "selection": "Lowest training-only five-fold CV Brier score; log loss breaks ties",
                "calibration": "Nested five-fold sigmoid calibration, ensemble=False",
                "schema_version": 2}
    (output / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    sns.set_theme(style="whitegrid", palette="deep")
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(bundle["test_y"], bundle["test_probability"] >= .5,
                                           labels=[0, 1], display_labels=["Fail", "Pass"], ax=ax, colorbar=False)
    ax.set_title("Held-out results · threshold 0.50")
    fig.tight_layout()
    fig.savefig(figures / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(bundle["test_y"], bundle["test_probability"], ax=ax, name="Selected Bayes")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    fig.tight_layout()
    fig.savefig(figures / "roc_curve.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Ideal")
    for label, p in [("Selected model", bundle["test_probability"]), ("Raw Bayes", bundle["raw_test_probability"])]:
        actual, predicted = calibration_curve(bundle["test_y"], p, n_bins=5, strategy="quantile")
        ax.plot(predicted, actual, marker="o", label=label)
    ax.set(xlabel="Mean predicted pass probability", ylabel="Observed pass fraction",
           xlim=(0, 1), ylim=(0, 1), title="Held-out calibration · 5 quantile bins")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "calibration_curve.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(bundle["statistics"]["spearman"], annot=True, fmt=".2f", vmin=-1, vmax=1,
                cmap="vlag", square=True, ax=ax)
    ax.set_title("Spearman correlation · training data")
    fig.tight_layout()
    fig.savefig(figures / "correlation.png", dpi=160)
    plt.close(fig)
    report = research_report(bundle)
    (output / "research_report.md").write_text(report, encoding="utf-8")


def research_report(bundle: dict) -> str:
    meta = bundle["metadata"]
    metrics = bundle["test_metrics"].iloc[0]
    return f"""# {TITLE}

## Abstract
This project estimates pass and fail probabilities using categorical Naive Bayes.
The target is explicitly defined as {meta['target_definition']}
The primary model is {bundle['model_name']}; it is selected by training-only cross-validation Brier score.
This version replaces the earlier dropout project. Dropout outcomes are not reused or renamed.

## Aim and objectives
Predict pass/fail probabilities, explain Bayes' theorem, analyze mean, sample standard deviation and correlation,
and let users change factors to observe prediction changes.
The four-factor CSV workflow uses attendance percentage, weekly study hours, previous marks and assignment scores.
The bundled real-data demonstration uses the closest verifiable UCI measurements and discloses the differences.

## Research questions
1. How are observed student factors associated with passing?
2. Does sigmoid calibration improve Naive Bayes probability quality on training folds?
3. How do probabilities change when one entered factor changes?
4. What data is needed to validate the four-factor model for a local institution?

## Dataset and target
Source: {meta['source']}.
Rows: {meta['rows']}; passes: {meta['pass_count']}; fails: {meta['fail_count']}.
Duplicates removed: {meta['duplicates_removed']}; missing predictor entries: {meta['missing_inputs']}.
SHA-256: {meta['sha256']}.
Schema: {bundle['schema']}. {SCHEMAS[bundle['schema']]['limitation']}
The official demonstration uses only mathematics records, avoiding cross-subject student overlap.
Its pass mark is a declared project convention of 10/20. Custom data defaults to 50/100, configurable before training.
No synthetic training records are generated. Final marks are used only to construct labels, never as predictors.

## Feature selection and timing
Predictors: {', '.join(SCHEMAS[bundle['schema']]['features'])}.
Gender, identifiers, final marks and other columns are excluded from model input.
G1 and G2 are prior-period grades, so the UCI model is a late-course prediction after G2, not an enrollment forecast.
The source does not establish an absence snapshot at that point: absences may include later information.
Consequently these are retrospective research results, not proof of prospective early-warning performance.
Local records must capture all predictors before the target assessment; assignment scores should be genuinely prior work.

## Probability methodology
Each input enters one of four predefined intervals. The study-time code is treated as categorical.
The model estimates a training prior P(Pass) and, for each factor, P(interval | Pass) and P(interval | Fail).
Laplace smoothing uses (class-and-interval count + 1)/(class count + 4).
Under conditional independence, J(c) = P(c) × product of P(interval_j | c).
Bayes' theorem gives P(Pass | inputs) = J(Pass)/(J(Pass) + J(Fail)).
P(Fail | inputs) = 1 - P(Pass | inputs).
Calculations use log space for numerical stability. The app reproduces every likelihood for an entered profile.
If calibrated Bayes is selected, sigmoid calibration adjusts the raw posterior; the app labels both separately.
The model is deliberately Bayesian; it is not selected against unrelated classifiers.
Fixed intervals improve interpretability but lose within-interval detail, so some slider changes leave the result unchanged.
Related grades violate the conditional independence approximation and can exaggerate certainty.

## Experimental design
Seed 42; stratified 80/20 split: {len(bundle['train_indices'])} training and {len(bundle['test_indices'])} testing rows.
Both raw Bayes and sigmoid-calibrated Bayes use five outer training folds.
Calibration uses five inner folds. Missing categories use most-frequent imputation fitted inside each fold.
Selection minimizes mean CV Brier score, with log loss as tie breaker, before touching test labels.
The final models remain fitted only to the training portion; the test set is not used for refitting.
A prevalence-only baseline uses the training pass rate.
The default decision threshold is 0.50; dashboard changes do not alter the fixed report metrics.

## Results
Selected model: {bundle['model_name']}.
Held-out accuracy: {metrics['accuracy']:.3f}; pass precision: {metrics['pass_precision']:.3f};
pass recall: {metrics['pass_recall']:.3f}; fail recall: {metrics['fail_recall']:.3f};
pass F1: {metrics['pass_f1']:.3f}; ROC-AUC: {metrics['roc_auc']:.3f};
pass average precision: {metrics['pass_average_precision']:.3f};
Brier score: {metrics['brier_score']:.3f}; log loss: {metrics['log_loss']:.3f}.
See model_comparison.csv for CV means and standard deviations, cv_fold_metrics.csv for every fold,
and test_metrics.csv for selected, raw and prior-only comparisons.
Calibration is measured rather than assumed; small held-out samples make reliability curves uncertain.

## Statistical analysis
All exploratory statistics use the training subset. summary.csv reports count, mean, sample standard deviation
(ddof=1), minimum, median and maximum. by_outcome.csv separates pass/fail groups.
Pearson and Spearman correlation matrices include binary pass (1) / fail (0).
The mean/std of ordinal study-time codes describe codes, not hours; Spearman is the preferred ordinal summary.
conditional.csv gives observed P(Pass | factor interval) with its denominator.
These single-factor observed frequencies differ from the smoothed model likelihood P(interval | class).
Missing observations are excluded pairwise from descriptive statistics, not filled with invented measurements.
Correlations and what-if curves show associations and model sensitivity, not causal effects.

## Ethics and limitations
The dataset is small, historical and from two Portuguese secondary schools. It does not establish validity
for universities or Pakistani institutions. Gender exclusion does not remove proxy bias.
No completed subgroup fairness audit is claimed in this revision.
Predicted probabilities are uncertain estimates, not guaranteed outcomes, intelligence or potential.
A user can receive a likely-fail label and still pass. Use human review and supportive action.
Uploads must be anonymous; the app processes custom records in session memory and does not write them to disk.
The hosting provider still operates the network infrastructure.
Custom data must use one row per student: repeated students require grouped splitting beyond this prototype.
Users must verify outcome completeness, source quality, collection time, sample size and local pass criteria.

## Conclusion and future work
The application demonstrates pass/fail Bayes prediction, transparent conditional probability and descriptive statistics.
Collect a representative, consented four-factor dataset for the intended institution, validate on a future cohort,
assess subgroup performance and probability calibration, and choose an operational threshold with educators.
Do not invent assignment or attendance measurements to make the UCI schema appear to match the proposal.

## References
- Cortez, P. (2008). Student Performance. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T (CC BY 4.0).
- Scikit-learn Naive Bayes: https://scikit-learn.org/stable/modules/naive_bayes.html
- Scikit-learn calibration: https://scikit-learn.org/stable/modules/calibration.html
- UNESCO Recommendation on the Ethics of AI: https://www.unesco.org/en/articles/recommendation-ethics-artificial-intelligence
"""
