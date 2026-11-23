"""Uncertainty of aggregate results, never individual student predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from src.config import SEED

BOOTSTRAP_RESAMPLES = 2000


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Two-sided 95% Wilson score interval; no observations means undefined."""
    if total < 0 or not 0 <= successes <= total:
        raise ValueError("Counts must satisfy 0 <= successes <= total.")
    if total == 0:
        return float("nan"), float("nan")
    z = norm.ppf(.975)
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / denominator
    return max(0., center - radius), min(1., center + radius)


def performance_intervals(y, probability, raw_probability,
                          resamples: int = BOOTSTRAP_RESAMPLES) -> pd.DataFrame:
    """Wilson classification intervals plus paired, class-stratified bootstrap.

    Resample held-out students within each observed class. Keep predictions fixed,
    and reuse the same rows for selected/raw model differences. This estimates
    test-sample uncertainty conditional on class counts and the fitted models;
    it excludes training, selection, school clustering and population drift.
    """
    y, p, raw = (np.asarray(a, dtype=float) for a in (y, probability, raw_probability))
    if (y.ndim != 1 or p.shape != y.shape or raw.shape != y.shape
            or set(np.unique(y)) != {0., 1.} or resamples < 100
            or not np.isfinite(p).all() or not np.isfinite(raw).all()
            or ((p < 0) | (p > 1) | (raw < 0) | (raw > 1)).any()):
        raise ValueError("Use aligned binary outcomes, valid probabilities, both classes and at least 100 resamples.")
    rows = []
    pred = p >= .5
    for metric, correct, total in [
        ("accuracy", int((pred == y).sum()), len(y)),
        ("pass_recall", int(pred[y == 1].sum()), int((y == 1).sum())),
        ("fail_recall", int((~pred[y == 0]).sum()), int((y == 0).sum())),
    ]:
        low, high = wilson_interval(correct, total)
        rows.append(dict(metric=metric, estimate=correct / total, lower_95=low,
                         upper_95=high, method="Wilson score", n=total, resamples=0))
    losses = (p - y)**2
    difference = losses - (raw - y)**2
    rng = np.random.default_rng(SEED)
    groups = [np.flatnonzero(y == c) for c in [0, 1]]
    sampled_loss, sampled_difference = [], []
    # Bound peak memory for uploaded datasets; do not retrain on test rows.
    for start in range(0, resamples, 100):
        size = min(100, resamples - start)
        ix = np.concatenate([rng.choice(group, (size, len(group)), replace=True)
                             for group in groups], axis=1)
        sampled_loss.extend(losses[ix].mean(axis=1))
        sampled_difference.extend(difference[ix].mean(axis=1))
    for metric, estimate, samples in [
        ("brier_score", losses.mean(), sampled_loss),
        ("brier_selected_minus_raw", difference.mean(), sampled_difference),
    ]:
        low, high = np.quantile(samples, [.025, .975])
        rows.append(dict(metric=metric, estimate=float(estimate), lower_95=float(low),
                         upper_95=float(high), method="Paired stratified percentile bootstrap",
                         n=len(y), resamples=resamples))
    return pd.DataFrame(rows)
