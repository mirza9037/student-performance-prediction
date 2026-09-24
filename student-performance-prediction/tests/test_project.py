"""Unit and real-artifact integration tests; run training before this suite."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.config import FIGURES, MODEL_PATH, REPORTS, ROOT, STAGES
from src.data import load_data, split_indices, validate_data
from src.evaluation import calculate_metrics, fairness_audit
from src.modeling import risk_band


@pytest.fixture(scope="module")
def data():
    return load_data()


@pytest.fixture(scope="module")
def bundle():
    assert MODEL_PATH.exists(), "Run python train.py before the integration tests."
    return joblib.load(MODEL_PATH)


def test_schema_rejects_invalid_data(data):
    frame = data[0].head(20).copy()
    with pytest.raises(ValueError, match="missing required"):
        validate_data(frame.drop(columns="Admission grade"))
    frame.loc[frame.index[0], "Target"] = "Unknown"
    with pytest.raises(ValueError, match="Unexpected target"):
        validate_data(frame)
    frame = data[0].head(20).copy()
    frame.loc[frame.index[0], "Admission grade"] = np.inf
    with pytest.raises(ValueError, match="Infinite"):
        validate_data(frame)


def test_official_labels_duplicates_and_whitespace(data, tmp_path):
    frame, y, meta = data
    assert len(frame) == 4424
    assert meta["source"].get("synthetic") is False
    np.testing.assert_array_equal(y, frame.Target.eq("Dropout").astype(int))
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    duplicated = duplicated.rename(columns={"Admission grade": " Admission grade "})
    path = tmp_path / "duplicate.csv"
    duplicated.to_csv(path, sep=";", index=False)
    clean, _, metadata = load_data(path)
    assert len(clean) == len(frame) and metadata["duplicates_removed"] == 1
    assert "Admission grade" in clean


def test_split_is_stratified_disjoint_and_reproducible(data):
    y = data[1]
    train, test = split_indices(y)
    again, _ = split_indices(y)
    assert set(train).isdisjoint(test)
    assert len(train) + len(test) == len(y)
    np.testing.assert_array_equal(train, again)
    assert abs(y.iloc[train].mean() - y.iloc[test].mean()) < .01


@pytest.mark.parametrize("separator", [",", ";"])
def test_both_official_csv_formats(data, tmp_path, separator):
    frame = data[0].copy()
    frame.columns = frame.columns.str.replace("semester (", "sem (", regex=False)
    path = tmp_path / "official_format.csv"
    frame.to_csv(path, sep=separator, index=False, encoding="utf-8-sig")
    loaded, _, _ = load_data(path)
    assert set(STAGES["first_semester"]).issubset(loaded.columns)
    assert len(loaded) == len(frame)


@pytest.mark.parametrize("stage", STAGES)
def test_saved_model_predictors_and_train_only_imputation(bundle, data, stage):
    frame, y, _ = data
    train, test = split_indices(y)
    result = bundle["stages"][stage]
    assert result["features"] == STAGES[stage]
    assert all("Gender" != name and "2nd semester" not in name for name in result["features"])
    if stage == "enrollment":
        assert not any("semester" in name for name in result["features"])
    pipeline = result["model"].calibrated_classifiers_[0].estimator
    transformer = pipeline.named_steps["preprocess"]
    assert transformer.transformers_[0][2] == STAGES[stage]
    medians = transformer.named_transformers_["numeric"].named_steps["imputer"].statistics_
    np.testing.assert_allclose(medians, frame.iloc[train][STAGES[stage]].median().to_numpy())
    sample = frame.iloc[test[:8]].copy()
    predicted = result["model"].predict_proba(sample)
    changed = sample.copy()
    changed["Gender"] = 1 - changed["Gender"]
    changed["Target"] = "Dropout"
    for column in changed.columns:
        if "2nd semester" in column:
            changed[column] = 999999
    np.testing.assert_allclose(predicted, result["model"].predict_proba(changed))
    assert np.isfinite(predicted).all() and (predicted >= 0).all() and (predicted <= 1).all()
    np.testing.assert_allclose(predicted.sum(axis=1), 1)
    sample.loc[sample.index[0], STAGES[stage][0]] = np.nan
    assert np.isfinite(result["model"].predict_proba(sample)).all()


def test_saved_test_metrics_match_actual_predictions(bundle, data):
    frame, y, _ = data
    _, test = split_indices(y)
    saved = pd.read_csv(REPORTS / "test_metrics.csv").set_index("stage")
    for stage, result in bundle["stages"].items():
        p = result["model"].predict_proba(frame.iloc[test])[:, 1]
        actual = calculate_metrics(y.iloc[test], p)
        for metric, value in actual.items():
            assert value == pytest.approx(saved.loc[stage, metric])
        assert sum(map(sum, result["confusion_matrix"])) == len(test)


def test_selection_uses_only_cv_and_all_models_are_compared(bundle):
    comparison = pd.read_csv(REPORTS / "model_comparison.csv")
    assert not any(name.startswith("test_") for name in comparison.columns)
    assert len(comparison) == 8
    assert len(pd.read_csv(REPORTS / "cv_fold_metrics.csv")) == 40
    for stage, group in comparison.groupby("stage"):
        best = group.sort_values(["cv_pr_auc", "cv_recall", "cv_brier_score", "cv_f1"], ascending=[False, False, True, False]).iloc[0]
        assert best["model"] == bundle["stages"][stage]["model_name"]
        assert group.selected.sum() == 1


def test_probability_metrics_and_threshold():
    y, p = np.array([0, 0, 1, 1]), np.array([.1, .4, .35, .8])
    scores = calculate_metrics(y, p)
    assert scores["brier_score"] == pytest.approx(np.mean((y - p) ** 2))
    assert scores["recall"] == .5
    lower = calculate_metrics(y, p, threshold=.30)
    assert lower["recall"] == 1
    assert scores["brier_score"] == lower["brier_score"]
    with pytest.raises(ValueError):
        calculate_metrics(y, np.array([np.nan, .4, .35, .8]))


@pytest.mark.parametrize("probability,expected", [(0, "Low"), (.2999, "Low"), (.3, "Medium"), (.5999, "Medium"), (.6, "High"), (1, "High")])
def test_risk_band_boundaries(probability, expected):
    assert risk_band(probability) == expected


def test_fairness_denominators_and_undefined_groups():
    y = pd.Series([1, 1, 0, 0, 1])
    p = np.array([.8, .2, .7, .1, .9])
    audit = fairness_audit(y, p, pd.Series([0, 0, 0, 0, 1]), "test").set_index("gender_code")
    assert audit.loc[0, "n_students"] == 4
    assert audit.loc[0, "recall"] == .5
    assert audit.loc[0, "false_positive_rate"] == .5
    assert audit.loc[0, "positive_prediction_rate"] == .5
    assert pd.isna(audit.loc[1, "false_positive_rate"])


def test_artifacts_are_complete(bundle):
    for name in ["research_report.md", "model_comparison.csv", "fairness_audit.csv", "feature_importance.csv",
                 "test_metrics.csv", "test_confidence_intervals.csv", "calibration_bins.csv", "experiment_metadata.json"]:
        assert (REPORTS / name).stat().st_size > 100
    for directory in [FIGURES, FIGURES / "enrollment", FIGURES / "first_semester"]:
        for name in ["confusion_matrix.png", "calibration_curve.png", "roc_curve.png", "feature_importance.png"]:
            assert (directory / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    meta = json.loads((REPORTS / "experiment_metadata.json").read_text(encoding="utf-8"))
    assert meta["sha256"] == bundle["metadata"]["sha256"]


@pytest.mark.parametrize("entrypoint", [ROOT / "app.py", ROOT.parent / "streamlit_app.py"])
def test_streamlit_all_pages_and_both_prediction_modes(entrypoint):
    app = AppTest.from_file(str(entrypoint), default_timeout=60).run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception and len(app.metric) == 2
    initial = app.metric[0].value
    app.slider[0].set_value(.30).run()
    assert not app.exception and app.metric[0].value == initial
    app.sidebar.selectbox[0].set_value("first_semester").run()
    app.button[0].click().run()
    assert not app.exception and len(app.metric) == 2
    for page in ["Model Performance", "Research Insights", "Ethics and Limitations"]:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, f"Exception in {page}: {app.exception}"
    app.sidebar.selectbox[0].set_value("enrollment").run()
    app.sidebar.radio[0].set_value("Model Performance").run()
    assert not app.exception


def test_streamlit_rejects_impossible_semester_counts():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    app.sidebar.selectbox[0].set_value("first_semester").run()
    for widget in app.number_input:
        if widget.label == "Curricular units 1st semester (enrolled)":
            widget.set_value(1)
        if widget.label == "Curricular units 1st semester (approved)":
            widget.set_value(2)
    app.button[0].click().run()
    assert not app.exception
    assert any("Approved curricular units" in error.value for error in app.error)
    assert not app.metric
