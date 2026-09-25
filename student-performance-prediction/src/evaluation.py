"""Evaluation with explicit class meaning and training-only model selection."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from src.config import SEED
from src.modeling import make_model, pass_probability


def calculate_metrics(y, probability) -> dict:
    pred = np.asarray(probability) >= .5
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "pass_precision": float(precision_score(y, pred, zero_division=0)),
        "pass_recall": float(recall_score(y, pred, zero_division=0)),
        "fail_recall": float(recall_score(1 - np.asarray(y), ~pred, zero_division=0)),
        "pass_f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "pass_average_precision": float(average_precision_score(y, probability)),
        "brier_score": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
    }


def split_indices(y):
    return train_test_split(np.arange(len(y)), test_size=.2, stratify=y, random_state=SEED)


def run_experiment(frame: pd.DataFrame, y: pd.Series, schema: str) -> dict:
    """Select raw/calibrated Bayes by CV Brier; test is evaluated afterwards."""
    train, test = split_indices(y)
    X_train, y_train = frame.iloc[train], y.iloc[train]
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows = []
    candidates = {"Naive Bayes": False, "Calibrated Naive Bayes": True}
    for name, calibrated in candidates.items():
        for fold, (fit, valid) in enumerate(cv.split(X_train, y_train), 1):
            model = make_model(schema, calibrated).fit(X_train.iloc[fit], y_train.iloc[fit])
            rows.append({"model": name, "fold": fold, **calculate_metrics(
                y_train.iloc[valid], pass_probability(model, X_train.iloc[valid]))})
    folds = pd.DataFrame(rows)
    comparison = folds.groupby("model").agg(
        cv_brier_mean=("brier_score", "mean"), cv_brier_std=("brier_score", "std"),
        cv_log_loss=("log_loss", "mean"), cv_roc_auc=("roc_auc", "mean"),
        cv_fail_recall=("fail_recall", "mean"), cv_accuracy=("accuracy", "mean"),
    ).sort_values(["cv_brier_mean", "cv_log_loss"]).reset_index()
    selected_name = comparison.iloc[0]["model"]
    raw = make_model(schema).fit(X_train, y_train)
    model = (make_model(schema, True).fit(X_train, y_train)
             if candidates[selected_name] else raw)
    probability = pass_probability(model, frame.iloc[test])
    raw_probability = pass_probability(raw, frame.iloc[test])
    baseline = np.full(len(test), y_train.mean())
    test_metrics = pd.DataFrame([
        {"model": "Selected Bayes model", **calculate_metrics(y.iloc[test], probability)},
        {"model": "Raw Bayes reference", **calculate_metrics(y.iloc[test], raw_probability)},
        {"model": "Training-prior baseline", **calculate_metrics(y.iloc[test], baseline)},
    ])
    return {"schema": schema, "model": model, "raw_model": raw, "model_name": selected_name,
            "train_indices": train, "test_indices": test, "fold_metrics": folds,
            "comparison": comparison, "test_metrics": test_metrics,
            "test_y": y.iloc[test].to_numpy(), "test_probability": probability,
            "raw_test_probability": raw_probability}
