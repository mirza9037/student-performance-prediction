"""Behavioral tests for Bayes, data contracts, evaluation and dashboard flows."""
from __future__ import annotations

import io
import json
import zipfile
import copy
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss
from streamlit.testing.v1 import AppTest
from src.config import DATA_PATH, MODEL_PATH, ROOT, SCHEMAS
from src.data import load_data, load_uploaded, validate_data
from src.evaluation import run_experiment, split_indices
from src.modeling import bayes_breakdown, classification, make_model, pass_probability, what_if
from src.reporting import statistical_analysis, evidence_archive
from src.uncertainty import wilson_interval, performance_intervals


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
    assert len(frame) == 649
    assert meta["pass_count"] == 549 and meta["fail_count"] == 100
    assert meta["course"] == "Portuguese language"
    np.testing.assert_array_equal(y, frame.G3.ge(10).astype(int))
    assert set(frame) == {"absences", "studytime", "G1", "G2", "G3"}
    assert "assignment_score" not in frame


def test_single_research_dataset_and_model_quality():
    frame, y, meta = load_data(DATA_PATH)
    bundle = joblib.load(MODEL_PATH)
    assert len(frame) == 649
    assert meta["pass_count"] == 549 and meta["fail_count"] == 100
    assert bundle["test_metrics"].iloc[0].accuracy > .90
    assert bundle["test_metrics"].iloc[0].brier_score < .08
    assert set(bundle["train_indices"]).isdisjoint(bundle["test_indices"])


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
    for page in ["Understand the probability", "Explore the data", "Check model quality", "Read the method", "Research evidence", "Predict"]:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, page
    assert len(app.sidebar.radio) == 1


def test_dashboard_custom_trained_session(custom_frame):
    frame, y, metadata = load_uploaded(custom_frame.to_csv(index=False).encode())
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
ns['research'](st.session_state['test_bundle'])
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


def test_wilson_intervals_boundaries_and_known_value():
    low, high = wilson_interval(50, 100)
    assert low == pytest.approx(.40383153)
    assert high == pytest.approx(.59616847)
    assert wilson_interval(0, 10)[0] == pytest.approx(0)
    assert wilson_interval(10, 10)[1] == pytest.approx(1)
    assert wilson_interval(0, 10)[1] > 0
    assert np.isnan(wilson_interval(0, 0)).all()
    with pytest.raises(ValueError):
        wilson_interval(11, 10)


def test_paired_intervals_reproducibility_and_identical_predictions(bundle):
    y, p = bundle['test_y'], bundle['test_probability']
    one = performance_intervals(y, p, p, resamples=200)
    pd.testing.assert_frame_equal(one, performance_intervals(y, p, p, resamples=200))
    paired = one.set_index('metric').loc['brier_selected_minus_raw']
    assert paired.estimate == paired.lower_95 == paired.upper_95 == 0
    assert one.set_index('metric').loc['brier_score', 'estimate'] == pytest.approx(brier_score_loss(y, p))
    assert one.set_index('metric').loc['fail_recall', 'n'] == (y == 0).sum()
    with pytest.raises(ValueError):
        performance_intervals([0, 0], [.1, .2], [.1, .2])


def test_paired_brier_direction_and_conditional_intervals(bundle):
    y = np.array([0, 1] * 20)
    table = performance_intervals(y, y, 1-y, resamples=100).set_index('metric')
    assert table.loc['brier_selected_minus_raw', 'upper_95'] == -1
    rows = bundle['statistics']['conditional']
    occupied = rows[rows.students > 0]
    assert (occupied.lower_95 <= occupied.observed_pass_rate + 1e-14).all()
    assert (occupied.upper_95 >= occupied.observed_pass_rate - 1e-14).all()
    assert rows[rows.students == 0][['lower_95', 'upper_95']].isna().all().all()


def test_evidence_export_active_dataset_no_student_rows(bundle, custom_frame):
    for active in [bundle]:
        with zipfile.ZipFile(io.BytesIO(evidence_archive(active))) as archive:
            assert 'confidence_intervals.csv' in archive.namelist()
            assert not any('model_bundle' in name or 'raw' in name for name in archive.namelist())
            metadata = json.loads(archive.read('experiment.json'))
            assert metadata['schema'] == active['schema']
            assert 'train_indices' not in metadata and 'test_y' not in metadata
            exported = pd.read_csv(io.BytesIO(archive.read('confidence_intervals.csv')))
            np.testing.assert_allclose(exported.estimate, active['intervals'].estimate)
    frame, y, meta = load_uploaded(custom_frame.to_csv(index=False).encode(), 60)
    custom = run_experiment(frame, y, 'custom')
    custom['metadata'] = meta
    ix = custom['train_indices']
    custom['statistics'] = statistical_analysis(frame.iloc[ix], y.iloc[ix], 'custom')
    with zipfile.ZipFile(io.BytesIO(evidence_archive(custom))) as archive:
        metadata = json.loads(archive.read('experiment.json'))
        assert metadata['pass_mark'] == 60
        assert metadata['features'] == SCHEMAS['custom']['features']
        assert 'Pass = final_marks >= 60' in archive.read('research_notes.md').decode()


def test_cloud_cache_refresh_and_legacy_bundle_upgrade(bundle):
    old = copy.deepcopy(bundle)
    old.pop('intervals')
    old['statistics']['conditional'] = old['statistics']['conditional'].drop(columns=['lower_95', 'upper_95'])
    first = io.BytesIO()
    joblib.dump(old, first)
    updated = copy.deepcopy(bundle)
    updated['metadata']['source'] = 'Updated artifact for cache regression test'
    second = io.BytesIO()
    joblib.dump(updated, second)
    script = """
import runpy
from pathlib import Path
import streamlit as st
ns = runpy.run_path(str(Path.cwd() / 'app.py'), run_name='cache_test')
old = ns['load_model_bytes'](st.session_state['first'])
assert 'intervals' in old
assert {'lower_95', 'upper_95'}.issubset(old['statistics']['conditional'].columns)
new = ns['load_model_bytes'](st.session_state['second'])
assert new['metadata']['source'] == 'Updated artifact for cache regression test'
assert old['metadata']['source'] != new['metadata']['source']
ns['research'](old)
"""
    app = AppTest.from_string(script, default_timeout=30)
    app.session_state['first'] = first.getvalue()
    app.session_state['second'] = second.getvalue()
    app.run()
    assert not app.exception
