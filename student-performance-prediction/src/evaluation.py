"""Probability metrics, group diagnostics, and reproducible research figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from src.config import SEED, THRESHOLD


def calculate_metrics(y: pd.Series | np.ndarray, probability: np.ndarray,
                      threshold: float = THRESHOLD) -> dict[str, float]:
    """PR-AUC is average precision (stepwise integral), not trapezoidal PR area."""
    y = np.asarray(y)
    probability = np.asarray(probability, dtype=float)
    if len(y) != len(probability) or not len(y):
        raise ValueError("Labels and probabilities must have the same nonzero length.")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Predicted probabilities must be finite and within [0, 1].")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be within [0, 1].")
    prediction = probability >= threshold
    two_classes = len(np.unique(y)) == 2
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)) if np.any(y == 1) else float("nan"),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)) if two_classes else float("nan"),
        "pr_auc": float(average_precision_score(y, probability)) if np.any(y == 1) else float("nan"),
        "brier_score": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
    }


def fairness_audit(y: pd.Series, probability: np.ndarray, gender: pd.Series,
                   stage: str, threshold: float = THRESHOLD) -> pd.DataFrame:
    """Audit held-out groups only; undefined rates remain NaN, never misleading zeroes."""
    frame = pd.DataFrame({"y": np.asarray(y), "p": probability,
                          "gender": gender.fillna(-1).to_numpy()})
    rows = []
    for group, subset in frame.groupby("gender"):
        truth, pred = subset["y"].to_numpy(), subset["p"].to_numpy() >= threshold
        tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0, 1]).ravel()
        rows.append({"stage": stage, "gender_code": int(group),
                     "gender_group": {0: "Female (source code 0)", 1: "Male (source code 1)"}.get(group, "Unknown"),
                     "n_students": len(subset), "n_positive": int(tp + fn), "n_negative": int(tn + fp),
                     "recall": tp / (tp + fn) if tp + fn else np.nan,
                     "false_positive_rate": fp / (fp + tn) if fp + tn else np.nan,
                     "positive_prediction_rate": float(pred.mean()),
                     "average_predicted_risk": float(subset["p"].mean()),
                     "observed_dropout_rate": float(subset["y"].mean()),
                     "brier_score": float(brier_score_loss(truth, subset["p"])), "threshold": threshold})
    return pd.DataFrame(rows)


def bootstrap_intervals(y: pd.Series, probability: np.ndarray, stage: str,
                        repetitions: int = 400) -> pd.DataFrame:
    """Descriptive 95% percentile intervals conditional on this fixed fitted model."""
    rng = np.random.default_rng(SEED)
    truth = np.asarray(y)
    scores = []
    for _ in range(repetitions):
        indices = rng.integers(0, len(truth), len(truth))
        if len(np.unique(truth[indices])) == 2:
            scores.append(calculate_metrics(truth[indices], probability[indices]))
    return pd.DataFrame([{"stage": stage, "metric": name,
                          "lower_95": np.quantile([score[name] for score in scores], .025),
                          "upper_95": np.quantile([score[name] for score in scores], .975)}
                         for name in scores[0]])


def calibration_bins(y: pd.Series, probability: np.ndarray, stage: str) -> pd.DataFrame:
    """Fixed probability bins with sample counts to expose sparse reliability estimates."""
    frame = pd.DataFrame({"observed_dropout_rate": np.asarray(y), "mean_predicted_risk": probability})
    frame["bin"] = pd.cut(probability, np.linspace(0, 1, 11), include_lowest=True)
    result = frame.groupby("bin", observed=True).agg(
        n_students=("observed_dropout_rate", "size"),
        observed_dropout_rate=("observed_dropout_rate", "mean"),
        mean_predicted_risk=("mean_predicted_risk", "mean")).reset_index()
    result["bin"] = result["bin"].astype(str)
    result.insert(0, "stage", stage)
    return result


def save_figures(y: pd.Series, probability: np.ndarray, raw_probability: np.ndarray,
                 importance: pd.DataFrame, output: Path, title: str) -> None:
    """Save all required plots, with labeled axes and the evaluation population."""
    output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    footer = "Source: official UCI 697 • held-out 20% test split • seed 42"

    def finish(name: str) -> None:
        plt.figtext(.5, .015, footer, ha="center", fontsize=8)
        plt.tight_layout(rect=(0, .05, 1, 1))
        plt.savefig(output / name, dpi=160, bbox_inches="tight")
        plt.close()

    plt.figure(figsize=(7, 5))
    sns.heatmap(confusion_matrix(y, probability >= THRESHOLD, labels=[0, 1]), annot=True, fmt="d", cmap="Blues",
                xticklabels=["Other (0)", "Dropout (1)"], yticklabels=["Other (0)", "Dropout (1)"], cbar=False)
    plt.xlabel("Predicted class (threshold 0.50)")
    plt.ylabel("Observed class")
    plt.title(f"{title}\nConfusion matrix (student counts)")
    finish("confusion_matrix.png")

    plt.figure(figsize=(7, 5))
    for values, label in [(raw_probability, "Before sigmoid"), (probability, "After sigmoid")]:
        observed, predicted = calibration_curve(y, values, n_bins=10, strategy="uniform")
        plt.plot(predicted, observed, marker="o", label=label)
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    plt.xlabel("Mean predicted dropout probability")
    plt.ylabel("Observed dropout fraction")
    plt.title(f"{title}\nReliability diagram (10 equal-width bins)")
    plt.legend()
    finish("calibration_curve.png")

    plt.figure(figsize=(7, 5))
    fpr, tpr, _ = roc_curve(y, probability)
    plt.plot(fpr, tpr, label=f"Calibrated model (AUC {roc_auc_score(y, probability):.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Chance ranking")
    plt.xlabel("False-positive rate")
    plt.ylabel("True-positive rate (recall)")
    plt.title(f"{title}\nROC curve")
    plt.legend()
    finish("roc_curve.png")

    top = importance.nlargest(10, "importance_mean").sort_values("importance_mean")
    plt.figure(figsize=(10, 6))
    plt.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#247b87")
    plt.axvline(0, color="gray", linewidth=.8)
    plt.xlabel("Decrease in average precision after permutation (mean ± SD; 10 repeats)")
    plt.ylabel("Predictor")
    plt.title(f"{title}\nTop ten global predictors; association, not causation")
    finish("feature_importance.png")
