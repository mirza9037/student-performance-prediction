"""Interactive, anonymous research dashboard. Launch with streamlit run app.py."""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import (BINARY_FEATURES, FIGURES, MODEL_PATH, REPORTS,
                        RESEARCH_QUESTIONS, STAGE_LABELS)
from src.modeling import risk_band

st.set_page_config(page_title="Student Risk Research", page_icon="🎓", layout="wide")


@st.cache_resource
def load_bundle(modified_at: float) -> dict:
    """Reload trusted local models when retraining changes the artifact timestamp."""
    del modified_at
    return joblib.load(MODEL_PATH)


def load_report(name: str) -> pd.DataFrame:
    """Explain how to recover missing training outputs without a traceback."""
    path = REPORTS / name
    if not path.exists():
        st.error(f"Missing {name}. Run python train.py from the project folder.")
        st.stop()
    return pd.read_csv(path)


def student_form(result: dict, stage: str) -> None:
    """Use source units, plausible constraints, and explicit submission for predictions."""
    st.subheader("Estimate recorded dropout risk")
    st.write("Enter an anonymous profile using the original dataset’s units. This model has not been validated for Pakistani students.")
    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01,
                          help="Only changes the support-review flag, not the predicted probability or risk band.")
    st.caption("Any operational threshold must be validated by the institution. Research default: 0.50.")
    values = {}
    with st.form(f"student_{stage}"):
        columns = st.columns(2, gap="large")
        for index, feature in enumerate(result["features"]):
            default = result["input_defaults"][feature]
            with columns[index % 2]:
                if feature in BINARY_FEATURES:
                    labels = {0: "Evening", 1: "Daytime"} if feature == "Daytime/evening attendance" else {0: "No", 1: "Yes"}
                    values[feature] = st.selectbox(feature, [0, 1], index=int(default >= .5), format_func=labels.get)
                else:
                    help_text = "Initial values are training-set medians; replace them with this student’s values."
                    if feature == "Application order":
                        values[feature] = st.number_input(feature, 0, 9, int(default), help="Source coding: 0 (first choice) to 9 (last choice).")
                    elif feature == "Age at enrollment":
                        values[feature] = st.number_input(feature, 0, 100, int(default), help=help_text)
                    elif "Curricular units" in feature and "(grade)" not in feature:
                        values[feature] = st.number_input(feature, 0, 500, int(default), help="Count, not a percentage. " + help_text)
                    elif feature in ["Previous qualification (grade)", "Admission grade"]:
                        values[feature] = st.number_input(feature + " (0–200)", 0.0, 200.0, float(default), step=0.1,
                                                         help="Portuguese source scale; do not enter a Pakistani percentage as an equivalent.")
                    elif "(grade)" in feature:
                        values[feature] = st.number_input(feature + " (0–20)", 0.0, 20.0, float(default), step=0.1,
                                                         help="Mean first-semester grade on the source scale.")
                    else:
                        label = feature + (" (source units)" if feature == "GDP" else " (%)")
                        help_text = ("UCI does not specify GDP units; preserve the source scale and verify local comparability."
                                     if feature == "GDP" else "Source macroeconomic percentage rate. Use a prediction-time observation.")
                        values[feature] = st.number_input(label, value=float(default), step=0.1, help=help_text)
        submitted = st.form_submit_button("Calculate risk probability", type="primary", use_container_width=True)
    if submitted:
        errors = []
        if stage == "first_semester":
            enrolled = values["Curricular units 1st semester (enrolled)"]
            approved = values["Curricular units 1st semester (approved)"]
            without = values["Curricular units 1st semester (without evaluations)"]
            if approved > enrolled:
                errors.append("Approved curricular units cannot exceed enrolled units.")
            if without > enrolled:
                errors.append("Units without evaluations cannot exceed enrolled units.")
        if errors:
            st.session_state.pop(f"prediction_{stage}", None)
            for error in errors:
                st.error(error)
        else:
            probability = float(result["model"].predict_proba(pd.DataFrame([values], columns=result["features"]))[0, 1])
            outside = [feature for feature, value in values.items()
                       if not result["input_ranges"][feature][0] <= value <= result["input_ranges"][feature][1]]
            st.session_state[f"prediction_{stage}"] = {"probability": probability, "outside": outside}
    prediction = st.session_state.get(f"prediction_{stage}")
    if prediction:
        probability = prediction["probability"]
        with st.container(border=True):
            left, right = st.columns([2, 1])
            left.metric("Estimated probability of recorded dropout", f"{probability:.1%}")
            right.metric("Risk band", risk_band(probability))
            st.progress(probability)
            st.write("**Support-review flag:** " + ("At or above threshold" if probability >= threshold else "Below threshold"))
            st.caption("Result from the last submitted profile. Recalculate after editing inputs. Low <30%; Medium 30–<60%; High ≥60%.")
            if prediction["outside"]:
                st.warning("Outside observed training ranges: " + ", ".join(prediction["outside"]) + ". This estimate may be unreliable.")
    st.info("Decision support only. This probability is not a final academic judgment and does not measure intelligence or potential. Human review is required before any intervention.")


def performance(result: dict, stage: str) -> None:
    st.subheader("Held-out model performance")
    st.caption("Selected using training CV. These test results and figures use the fixed 0.50 threshold.")
    columns = st.columns(4)
    for column, key, label in zip(columns, ["pr_auc", "recall", "brier_score", "roc_auc"],
                                   ["PR-AUC (average precision)", "At-risk recall", "Brier score ↓", "ROC-AUC"]):
        column.metric(label, f"{result['metrics'][key]:.3f}")
    st.write(f"Selected model: **{result['model_name']}**")
    st.dataframe(load_report("test_metrics.csv"), hide_index=True, use_container_width=True)
    st.subheader("Training-only model comparison")
    st.caption("Five outer folds with five inner calibration folds. Rank: AP → recall → lower Brier → F1. Full CSV includes fold standard deviations.")
    comparison = load_report("model_comparison.csv")
    st.dataframe(comparison[["stage", "model", "cv_pr_auc", "cv_recall", "cv_brier_score", "cv_f1", "selected"]],
                 hide_index=True, use_container_width=True)
    st.download_button("Download full comparison CSV", comparison.to_csv(index=False), "model_comparison.csv", "text/csv")
    plots = st.columns(2)
    for index, name in enumerate(["confusion_matrix.png", "calibration_curve.png", "roc_curve.png", "feature_importance.png"]):
        path = FIGURES / stage / name
        if path.exists():
            plots[index % 2].image(str(path), use_container_width=True)
        else:
            plots[index % 2].warning(f"Missing {name}; run python train.py.")
    with st.expander("How to read the metrics", expanded=True):
        st.markdown("""- **Precision:** how many flagged students actually have the dropout label.
- **Recall:** how many recorded dropouts are detected.
- **F1:** a balance between precision and recall.
- **PR-AUC (average precision):** how well the ranking retrieves dropouts; higher is better.
- **ROC-AUC:** how often a dropout ranks above a non-dropout; higher is better.
- **Brier score:** average squared probability error; lower is better.
- **Log loss:** penalizes confident wrong probabilities; lower is better.
- **Accuracy:** fraction correctly classified; secondary because classes are imbalanced.
- **Calibration curve:** compares predicted probabilities with observed dropout fractions.
- **Confusion matrix:** counts correct decisions, missed dropouts, and false alerts.""")
    st.subheader("Gender audit · test set · threshold 0.50")
    audit = load_report("fairness_audit.csv")
    st.dataframe(audit[audit.stage == stage], hide_index=True, use_container_width=True)
    st.caption("Gender is never a prediction input. Group differences need investigation; they do not automatically prove discrimination. Small groups give uncertain estimates.")
    with st.expander("Test uncertainty and calibration-bin counts"):
        ci = load_report("test_confidence_intervals.csv")
        st.dataframe(ci[ci.stage == stage], hide_index=True)
        st.caption("95% percentile bootstrap intervals, 400 resamples. Conditional on this fitted model and split; excludes training variability and distribution shift.")
        bins = load_report("calibration_bins.csv")
        st.dataframe(bins[bins.stage == stage], hide_index=True)


def insights(bundle: dict, stage: str) -> None:
    st.subheader("Research design and findings")
    meta = bundle["metadata"]
    st.write(f"Official UCI Portuguese higher-education dataset: **{meta['rows']:,} students** after removing {meta['duplicates_removed']} exact duplicates. "
             f"Shared split: **{meta['train_size']:,} training / {meta['test_size']:,} test**. No synthetic data.")
    st.markdown("[Dataset and attribution · UCI, CC BY 4.0](https://doi.org/10.24432/C5MC89)")
    counts = pd.DataFrame({"Outcome": list(meta["original_target_counts"]), "Students": list(meta["original_target_counts"].values())})
    chart = px.bar(counts, x="Outcome", y="Students", title="Original target-class distribution", color_discrete_sequence=["#247b87"])
    st.plotly_chart(chart, use_container_width=True)
    st.caption("Source: complete deduplicated UCI dataset. Dropout = 1; Graduate or Enrolled = 0. Enrolled is an unresolved outcome.")
    for index, question in enumerate(RESEARCH_QUESTIONS, 1):
        st.write(f"{index}. {question}")
    results = load_report("test_metrics.csv")
    chart_data = results.melt(id_vars="stage", value_vars=["pr_auc", "recall", "brier_score"], var_name="Metric", value_name="Score")
    fig = px.bar(chart_data, x="Metric", y="Score", color="stage", barmode="group", range_y=[0, 1],
                 title="Enrollment vs first-semester test results", labels={"stage": "Prediction stage", "Score": "Metric score (0–1)"},
                 color_discrete_sequence=["#247b87", "#8392a5"])
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Source: shared held-out UCI test split. Higher AP/recall is better; lower Brier is better. Recall uses threshold 0.50.")
    st.subheader("Top ten global predictors")
    importance = load_report("feature_importance.csv")
    st.dataframe(importance[importance.stage == stage].head(10), hide_index=True, use_container_width=True)
    st.caption("Decrease in test average precision after shuffling a feature. Mean and SD over 10 repeats. Correlated features can mask importance. Association does not prove causation.")
    st.subheader("What calibration means")
    st.write("If probabilities are well calibrated in a population, roughly 70% of students assigned risk near 70% have the recorded dropout outcome. Sigmoid calibration learns a correction using training-only out-of-fold scores. It does not guarantee accurate probabilities after a change of institution or cohort.")
    st.warning("Timing limitation: debtor, tuition and scholarship status must be measured at enrollment for enrollment prediction. The dataset does not supply per-field timestamps. Verify timing before local use.")
    report = REPORTS / "research_report.md"
    if report.exists():
        st.download_button("Download full research report", report.read_text(encoding="utf-8"), "research_report.md", "text/markdown")


def ethics() -> None:
    st.subheader("Responsible use requires local evidence")
    st.markdown("""**Privacy.** Enter no names, student IDs, contact details, or other identifying information. This prototype does not write submitted profiles to disk. Inputs remain temporarily in the active session. Institutional authentication, access control, consent and retention policies are required before deployment.

**Bias and fairness.** Gender is excluded as an input but retained for evaluation. Other variables may carry proxy information. Review error rates, probability reliability, sample sizes and potential harms with affected students. Binary source gender coding is incomplete.

**Data drift.** Changes in admissions, grading, fees, economic conditions or support programs can change risk relationships. Monitor input distributions, dropout prevalence and calibration on newly observed outcomes. Revalidate before retraining or changing thresholds.

**Portugal to Pakistan.** Educational systems, marks, tuition, enrollment policies and social conditions differ. These results do not establish validity for Pakistani students. Gather time-stamped local data, align outcome definitions and scales, validate on held-out local cohorts, and retrain or recalibrate as necessary.

**Human review.** A qualified person must review the context before every intervention. Offer supportive help, allow correction and appeal, and never automate disciplinary action, admissions exclusion or denial of funding from this score.

**Limits of the outcome.** Dropout is not a measure of intelligence, character or potential. Enrolled students have unresolved outcomes. Predicted probabilities are uncertain and feature importance is not causal evidence.""")
    st.markdown("[UNESCO Recommendation on the Ethics of AI](https://www.unesco.org/en/articles/recommendation-ethics-artificial-intelligence)")


def main() -> None:
    st.title("Student Performance Prediction")
    st.caption("AI-BASED RESEARCH · CALIBRATED DROPOUT PROBABILITIES · HUMAN REVIEW")
    if not MODEL_PATH.exists():
        st.error("Trained model missing. Open a terminal in this project and run python train.py, then reload.")
        st.stop()
    try:
        bundle = load_bundle(MODEL_PATH.stat().st_mtime)
    except Exception as exc:
        st.error(f"Cannot load the local model bundle ({type(exc).__name__}). Run python train.py in this environment to regenerate it.")
        st.stop()
    with st.sidebar:
        st.header("Research workspace")
        page = st.radio("Page", ["Individual Prediction", "Model Performance", "Research Insights", "Ethics and Limitations"])
        stage = st.selectbox("Prediction stage", list(STAGE_LABELS), format_func=STAGE_LABELS.get)
        st.divider()
        st.write("**Research prototype**")
        st.caption("Anonymous inputs · no automatic interventions")
        st.caption("Seed 42 · sigmoid calibration · 5-fold CV")
    result = bundle["stages"][stage]
    if page == "Individual Prediction":
        student_form(result, stage)
    elif page == "Model Performance":
        performance(result, stage)
    elif page == "Research Insights":
        insights(bundle, stage)
    else:
        ethics()


if __name__ == "__main__":
    main()
