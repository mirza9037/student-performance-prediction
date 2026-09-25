"""Behavioral tests for Bayes, data contracts, evaluation and dashboard flows."""
from __future__ import annotations

import io
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss
from streamlit.testing.v1 import AppTest
from src.config import MODEL_PATH, ROOT, SCHEMAS
from src.data import load_data, load_uploaded, validate_data
from src.evaluation import run_experiment, split_indices
from src.modeling import bayes_breakdown, classification, make_model, pass_probability, what_if
from src.reporting import statistical_analysis


@pytest.fixture(scope="module")
def data():
    return load_data()


@pytest.fixture(scope="module")
def bundle():
    return joblib.load(MODEL_PATH)


@pytest.fixture
def custom_frame():
    # Test fixtures only, never shipped as a student dataset or research results.
    rng = np.random.default_rng(9)
    n = 100
    return pd.DataFrame({"attendance": rng.uniform(20, 100, n),
                         "study_hours": rng.uniform(0, 40, n),
                         "previous_marks": rng.uniform(0, 100, n),
                         "assignment_score": rng.uniform(0, 100, n),
                         "final_marks": np.tile([35, 65], n // 2)})


def test_real_dataset_target_and_no_relabeling(data):
    frame, y, meta = data
    assert len(frame) == 395
    assert meta["pass_count"] == 265 and meta["fail_count"] == 130
    np.testing.assert_array_equal(y, frame.G3.ge(10).astype(int))
    assert set(frame) == {"absences", "studytime", "G1", "G2", "G3"}
    assert "assignment_score" not in frame


@pytest.mark.parametrize("field,value,match", [
    ("attendance", -1, "between"), ("study_hours", 169, "between"),
    ("assignment_score", np.inf, "Infinite"), ("final_marks", np.nan, "cannot be missing"),
    ("previous_marks", "invalid", "must contain numbers"),
])
def test_invalid_custom_data(custom_frame, field, value, match):
    custom_frame[field] = custom_frame[field].astype(object)
    custom_frame.loc[0, field] = value
    with pytest.raises(ValueError, match=match):
        validate_data(custom_frame, "custom")


def test_missing_schema_and_all_missing(custom_frame):
    with pytest.raises(ValueError, match="Missing required"):
        validate_data(custom_frame.drop(columns="assignment_score"), "custom")
    custom_frame["attendance"] = np.nan
    with pytest.raises(ValueError, match="entirely missing"):
        validate_data(custom_frame, "custom")


def test_custom_labels_pass_mark_duplicates_and_identifiers(custom_frame):
    custom_frame["name"] = "excluded"
    source = pd.concat([custom_frame, custom_frame.iloc[[0]]], ignore_index=True)
    frame, y, meta = validate_data(source, "custom", 65)
    assert len(frame) == 100 and meta["duplicates_removed"] == 1
    assert "name" not in frame and y.sum() == 50
    with pytest.raises(ValueError, match="15 passes"):
        validate_data(source, "custom", 70)
    with pytest.raises(ValueError, match="Pass mark"):
        validate_data(source, "custom", 0)


def test_custom_upload_and_missing_input(custom_frame):
    custom_frame.loc[0, "attendance"] = np.nan
    frame, y, metadata = load_uploaded(custom_frame.to_csv(index=False).encode())
    assert metadata["missing_inputs"] == 1
    model = make_model("custom").fit(frame, y)
    assert np.isfinite(pass_probability(model, frame)).all()
    with pytest.raises(ValueError, match="5 MB"):
        load_uploaded(b" " * (5 * 1024 * 1024 + 1))


def test_split_is_disjoint_stratified_reproducible(data, bundle):
    _, y, _ = data
    train, test = split_indices(y)
    assert set(train).isdisjoint(test)
    np.testing.assert_array_equal(train, bundle["train_indices"])
    np.testing.assert_array_equal(test, bundle["test_indices"])
    assert abs(y.iloc[train].mean() - y.iloc[test].mean()) < .01


def test_bayes_formula_matches_raw_model_and_smoothing(data, bundle):
    frame = data[0]
    for i in [0, 17, 150, 394]:
        explanation = bayes_breakdown(bundle["raw_model"], frame.iloc[[i]], "uci")
        expected = bundle["raw_model"].predict_proba(frame.iloc[[i]])[0]
        np.testing.assert_allclose(explanation["posterior"], expected, atol=1e-12)
        np.testing.assert_allclose(explanation["posterior"].sum(), 1)
        assert (explanation["terms"].iloc[:, 2:].to_numpy() > 0).all()
    nb = bundle["raw_model"].named_steps["bayes"]
    for counts, logp in zip(nb.category_count_, nb.feature_log_prob_):
        np.testing.assert_allclose(np.exp(logp), (counts + 1) / (nb.class_count_[:, None] + 4))


def test_model_never_uses_final_marks_or_gender(data, bundle):
    frame = data[0].iloc[:20].copy()
    before = pass_probability(bundle["model"], frame)
    frame["G3"] = 0
    frame["Gender"] = 999
    np.testing.assert_allclose(before, pass_probability(bundle["model"], frame))
    assert list(bundle["raw_model"].named_steps["bins"].feature_names_in_) == SCHEMAS["uci"]["features"]
    nb = bundle["raw_model"].named_steps["bayes"]
    assert nb.class_count_.sum() == len(bundle["train_indices"])


def test_metrics_and_cv_selection(bundle):
    comparison = bundle["comparison"]
    assert bundle["model_name"] == comparison.sort_values(["cv_brier_mean", "cv_log_loss"]).iloc[0]["model"]
    assert len(bundle["fold_metrics"]) == 10
    actual = brier_score_loss(bundle["test_y"], bundle["test_probability"])
    assert actual == pytest.approx(bundle["test_metrics"].iloc[0]["brier_score"])
    assert np.isfinite(bundle["test_metrics"].iloc[:, 1:].to_numpy()).all()


def test_boundaries_and_what_if(data, bundle):
    assert classification(.5) == "Likely pass"
    assert classification(.49) == "Likely fail"
    assert classification(.6, .7) == "Likely fail"
    row = data[0].iloc[[0]].copy()
    curve = what_if(bundle["model"], row, "uci", "G2")
    assert curve.iloc[:, 1].nunique() > 1
    assert curve.iloc[:, 1].between(0, 1).all()
    row["G1"] = 30
    with pytest.raises(ValueError, match="between"):
        pass_probability(bundle["model"], row)


def test_statistics_use_training_records_and_correct_denominators(data, bundle):
    frame, y, _ = data
    ix = bundle["train_indices"]
    stats = statistical_analysis(frame.iloc[ix], y.iloc[ix], "uci")
    assert stats["summary"].loc["G1", "mean"] == pytest.approx(frame.iloc[ix].G1.mean())
    assert stats["summary"].loc["G1", "std"] == pytest.approx(frame.iloc[ix].G1.std(ddof=1))
    for feature, groups in stats["conditional"].groupby("feature"):
        assert groups.students.sum() == len(ix)
        occupied = groups[groups.students > 0]
        np.testing.assert_allclose(occupied.observed_pass_rate, occupied.passes / occupied.students)


def test_custom_full_training_and_all_four_predictors(custom_frame):
    frame, y, meta = validate_data(custom_frame, "custom")
    result = run_experiment(frame, y, "custom")
    assert len(result["test_indices"]) == 20
    assert result["raw_model"].named_steps["bayes"].n_features_in_ == 4
    raw = bayes_breakdown(result["raw_model"], frame.iloc[[0]], "custom")
    np.testing.assert_allclose(raw["posterior"], result["raw_model"].predict_proba(frame.iloc[[0]])[0])


@pytest.mark.parametrize("entry", [ROOT / "app.py", ROOT.parent / "streamlit_app.py"])
def test_dashboard_pages_inputs_and_threshold(entry):
    app = AppTest.from_file(str(entry), default_timeout=30).run()
    assert not app.exception
    original = app.session_state["last_pass_probability"]
    app.slider(key="uci_G2").set_value(0).run()
    assert not app.exception
    changed = app.session_state["last_pass_probability"]
    assert original != changed
    app.slider(key="uci_threshold").set_value(.8).run()
    assert app.session_state["last_pass_probability"] == changed
    for page in ["Bayes explained", "Statistics", "Model performance", "Data and method", "Prediction"]:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, page
    app.sidebar.radio(key="source").set_value("My four-factor CSV").run()
    assert not app.exception
    assert any("attendance" in text.value for text in app.markdown)
    assert not app.metric


def test_dashboard_custom_trained_session(custom_frame):
    frame, y, metadata = validate_data(custom_frame, "custom")
    result = run_experiment(frame, y, "custom")
    result["metadata"] = metadata
    ix = result["train_indices"]
    result["statistics"] = statistical_analysis(frame.iloc[ix], y.iloc[ix], "custom")
    # Exercise actual UI page functions with a trained custom bundle.
    script = """
import runpy
from pathlib import Path
import streamlit as st
ns = runpy.run_path(str(Path.cwd() / 'app.py'), run_name='test_dashboard')
ns['prediction'](st.session_state['test_bundle'])
ns['explain_bayes'](st.session_state['test_bundle'])
ns['statistics'](st.session_state['test_bundle'])
ns['performance'](st.session_state['test_bundle'])
"""
    app = AppTest.from_string(script, default_timeout=30)
    app.session_state["test_bundle"] = result
    app.run()
    assert not app.exception
    assert len([s for s in app.slider if s.key and s.key.startswith("custom_")]) == 5
    p = app.session_state["last_pass_probability"]
    app.slider(key="custom_assignment_score").set_value(5).run()
    assert not app.exception
    assert 0 <= app.session_state["last_pass_probability"] <= 1


def test_research_artifacts_current(bundle):
    for name in ["summary.csv", "conditional.csv", "spearman.csv", "pearson.csv",
                 "test_metrics.csv", "model_comparison.csv", "research_report.md"]:
        assert (ROOT / "reports" / name).exists()
    text = (ROOT / "reports" / "research_report.md").read_text(encoding="utf-8")
    assert "Pass = G3 >= 10" in text
    assert bundle["schema_version"] == 2
