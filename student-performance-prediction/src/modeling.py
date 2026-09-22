"""Training-only model comparison with nested probability calibration."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import FOLDS, SEED
from src.evaluation import calculate_metrics

MODEL_NAMES = ["Logistic Regression", "Gaussian Naive Bayes", "Random Forest", "HistGradientBoosting"]


def make_model(name: str, features: list[str]) -> CalibratedClassifierCV:
    """Wrap the entire preprocessing pipeline in inner, stratified 5-fold calibration."""
    estimators = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED),
        "Gaussian Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=120, min_samples_leaf=3, class_weight="balanced", n_jobs=1, random_state=SEED),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15, l2_regularization=1.0,
                                                              class_weight="balanced", early_stopping=False, random_state=SEED),
    }
    if name not in estimators:
        raise ValueError(f"Unknown model: {name}")
    steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if name in {"Logistic Regression", "Gaussian Naive Bayes"}:
        steps.append(("scaler", StandardScaler()))
    preprocess = ColumnTransformer([("numeric", Pipeline(steps), features)], remainder="drop")
    pipeline = Pipeline([("preprocess", preprocess), ("classifier", estimators[name])])
    # ensemble=False obtains out-of-fold calibration scores, then refits the base
    # pipeline on its full training partition. No outer validation/test row is used.
    return CalibratedClassifierCV(pipeline, method="sigmoid", ensemble=False,
                                  cv=StratifiedKFold(FOLDS, shuffle=True, random_state=SEED), n_jobs=1)


def compare_models(X: pd.DataFrame, y: pd.Series, features: list[str], stage: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Evaluate candidates in outer CV; select lexicographically by AP, recall, Brier, F1."""
    rows, fold_rows = [], []
    outer = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
    for name in MODEL_NAMES:
        calibrated_scores, raw_scores = [], []
        for fold, (fit, valid) in enumerate(outer.split(X, y), start=1):
            model = make_model(name, features)
            model.fit(X.iloc[fit], y.iloc[fit])
            probability = model.predict_proba(X.iloc[valid])[:, 1]
            raw_probability = model.calibrated_classifiers_[0].estimator.predict_proba(X.iloc[valid])[:, 1]
            scores = calculate_metrics(y.iloc[valid], probability)
            raw = calculate_metrics(y.iloc[valid], raw_probability)
            calibrated_scores.append(scores)
            raw_scores.append(raw)
            fold_rows.append({"stage": stage, "model": name, "fold": fold, **scores,
                              "raw_brier_score": raw["brier_score"], "raw_log_loss": raw["log_loss"]})
            print(f"  {stage} | {name} | fold {fold}/{FOLDS} | AP={scores['pr_auc']:.4f}", flush=True)
        row = {"stage": stage, "model": name}
        for metric in calibrated_scores[0]:
            values = [score[metric] for score in calibrated_scores]
            row[f"cv_{metric}"] = float(np.mean(values))
            row[f"cv_{metric}_std"] = float(np.std(values, ddof=1))
        row["cv_raw_brier_score"] = float(np.mean([score["brier_score"] for score in raw_scores]))
        row["cv_raw_log_loss"] = float(np.mean([score["log_loss"] for score in raw_scores]))
        rows.append(row)
    comparison = pd.DataFrame(rows).sort_values(
        ["cv_pr_auc", "cv_recall", "cv_brier_score", "cv_f1"], ascending=[False, False, True, False], kind="stable"
    ).reset_index(drop=True)
    comparison["selected"] = comparison.index == 0
    return comparison, pd.DataFrame(fold_rows), str(comparison.iloc[0]["model"])


def risk_band(probability: float) -> str:
    """Descriptive probability bands, independent of the operational threshold."""
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Risk probability must be a finite number between 0 and 1.")
    return "Low" if probability < 0.30 else "Medium" if probability < 0.60 else "High"
