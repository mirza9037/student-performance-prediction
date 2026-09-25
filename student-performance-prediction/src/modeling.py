"""Interpretable Naive Bayes with fixed intervals and optional calibration."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import CategoricalNB
from sklearn.pipeline import Pipeline
from src.config import SCHEMAS, SEED


class FixedBins(TransformerMixin, BaseEstimator):
    """Select the feature allowlist, validate values, and assign fixed intervals."""

    def __init__(self, schema: str = "uci"):
        self.schema = schema

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = np.array(SCHEMAS[self.schema]["features"], dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        spec = SCHEMAS[self.schema]
        values = X.loc[:, spec["features"]].to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError("Infinite input values are not allowed.")
        for j, (lo, hi) in enumerate(spec["bounds"]):
            finite = values[:, j][~np.isnan(values[:, j])]
            if ((finite < lo) | (finite > hi)).any():
                raise ValueError(f"{spec['features'][j]} must be between {lo} and {hi}.")
            if self.schema == "uci" and (finite != np.floor(finite)).any():
                raise ValueError("UCI inputs must be whole numbers or valid study-time bands.")
        return np.column_stack([
            np.where(np.isnan(values[:, j]), np.nan, np.digitize(values[:, j], edges))
            for j, edges in enumerate(spec["bins"])
        ])


def make_model(schema: str, calibrated: bool = False):
    """Impute within each fold; Laplace smoothing keeps empty bins possible."""
    model = Pipeline([
        ("bins", FixedBins(schema)),
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("bayes", CategoricalNB(alpha=1.0, min_categories=[4] * 4)),
    ])
    if calibrated:
        return CalibratedClassifierCV(
            model, method="sigmoid", ensemble=False,
            cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=1,
        )
    return model


def pass_probability(model, X: pd.DataFrame) -> np.ndarray:
    """Read the positive-class column explicitly: 1 always means pass."""
    return model.predict_proba(X)[:, list(model.classes_).index(1)]


def classification(probability: float, threshold: float = 0.5) -> str:
    return "Likely pass" if probability >= threshold else "Likely fail"


def bin_label(edges: list, index: int) -> str:
    if index == 0:
        return f"< {edges[0]}"
    if index == len(edges):
        return f"≥ {edges[-1]}"
    return f"{edges[index - 1]} to < {edges[index]}"


def bayes_breakdown(raw_model: Pipeline, row: pd.DataFrame, schema: str) -> dict:
    """Reconstruct the exact raw posterior from priors and conditional counts."""
    categories = raw_model[:-1].transform(row).astype(int)[0]
    nb = raw_model.named_steps["bayes"]
    terms = []
    log_joint = nb.class_log_prior_.copy()
    for j, category in enumerate(categories):
        log_likelihood = nb.feature_log_prob_[j][:, category]
        log_joint += log_likelihood
        terms.append({
            "Factor": SCHEMAS[schema]["labels"][j],
            "Interval": bin_label(SCHEMAS[schema]["bins"][j], category),
            "P(interval | Fail)": float(np.exp(log_likelihood[0])),
            "P(interval | Pass)": float(np.exp(log_likelihood[1])),
        })
    posterior = np.exp(log_joint - logsumexp(log_joint))
    return {"terms": pd.DataFrame(terms), "priors": np.exp(nb.class_log_prior_),
            "joint": np.exp(log_joint), "posterior": posterior}


def what_if(model, row: pd.DataFrame, schema: str, feature: str) -> pd.DataFrame:
    """Vary one input while holding all other entered values fixed."""
    spec = SCHEMAS[schema]
    j = spec["features"].index(feature)
    lo, hi = spec["bounds"][j]
    grid = np.arange(lo, hi + 1, dtype=float)
    scenarios = pd.concat([row] * len(grid), ignore_index=True)
    scenarios[feature] = grid
    return pd.DataFrame({spec["labels"][j]: grid,
                         "Pass probability": pass_probability(model, scenarios)})
