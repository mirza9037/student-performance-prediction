"""Interactive pass/fail probability dashboard."""
from __future__ import annotations

import hashlib
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, roc_curve
from threadpoolctl import threadpool_limits
from src.config import MODEL_PATH, SCHEMAS, STUDY_BANDS, TITLE
from src.data import load_uploaded
from src.evaluation import run_experiment
from src.modeling import bayes_breakdown, classification, pass_probability, what_if
from src.reporting import statistical_analysis

st.set_page_config(page_title="Student Performance · Bayes", page_icon="🎓", layout="wide")
st.markdown("""
<style>
.block-container { max-width: 1200px; padding-top: 2rem; }
[data-testid="stMetric"] { border: 1px solid #dae4ed; padding: 1rem; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_builtin():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model missing. Run python train.py, then restart the app.")
    bundle = joblib.load(MODEL_PATH)
    if bundle.get("schema_version") != 2:
        raise ValueError("The saved model belongs to the old project. Run python train.py.")
    return bundle


def choose_data():
    source = st.sidebar.radio("Dataset", ["Research dataset", "My four-factor CSV"], key="source")
    if source == "Research dataset":
        return load_builtin()
    st.sidebar.caption("Anonymous data stays in this app session; it is not saved to disk.")
    st.subheader("Use attendance, study hours, previous marks and assignments")
    st.write("Upload historical records with final results to train your own pass/fail model. "
             "Use one row per student and measure all inputs before the final assessment.")
    st.download_button("Download CSV headers",
                       "attendance,study_hours,previous_marks,assignment_score,final_marks\n",
                       file_name="student_data_template.csv", mime="text/csv")
    with st.expander("CSV format and requirements", expanded=True):
        st.markdown(
            "- attendance: percentage, 0–100\n"
            "- study_hours: hours per week, 0–168\n"
            "- previous_marks: percentage, 0–100\n"
            "- assignment_score: percentage, 0–100\n"
            "- final_marks: final assessment percentage, 0–100\n\n"
            "Use 50–10,000 rows with at least 15 passes and 15 fails. "
            "Blank predictor cells are handled during training; final_marks must be present. "
            "A larger representative sample is needed for useful research conclusions. "
            "Do not include names, emails or student IDs.")
    pass_mark = st.number_input("Actual pass mark (%)", min_value=1.0, max_value=100.0,
                                value=50.0, step=1.0, key="custom_pass_mark")
    uploaded = st.file_uploader("Anonymous student CSV (maximum 5 MB)", type=["csv"])
    if uploaded is None:
        st.info("The four requested inputs become available after a valid dataset is trained. "
                "The Research dataset option works immediately with verified UCI records.")
        return None
    payload = uploaded.getvalue()
    fingerprint = hashlib.sha256(payload + str(pass_mark).encode()).hexdigest()
    if st.session_state.get("custom_fingerprint") != fingerprint:
        st.session_state.pop("custom_bundle", None)
    if st.button("Train my Bayes model", type="primary"):
        try:
            frame, y, metadata = load_uploaded(payload, pass_mark)
            with st.spinner("Training Bayes models and evaluating held-out records..."):
                with threadpool_limits(limits=2):
                    bundle = run_experiment(frame, y, "custom")
                bundle["metadata"] = metadata
                ix = bundle["train_indices"]
                bundle["statistics"] = statistical_analysis(frame.iloc[ix], y.iloc[ix], "custom")
                st.session_state["custom_bundle"] = bundle
                st.session_state["custom_fingerprint"] = fingerprint
        except (ValueError, KeyError) as exc:
            st.error(str(exc))
            return None
    return st.session_state.get("custom_bundle")


def prediction(bundle):
    schema = bundle["schema"]
    spec = SCHEMAS[schema]
    st.header("Explore a student's pass probability")
    st.caption("Change an input to recalculate immediately. No name or student ID is needed.")
    st.info(spec["limitation"])
    left, right = st.columns([1, 1.15], gap="large")
    values = {}
    with left:
        for i, feature in enumerate(spec["features"]):
            lo, hi = spec["bounds"][i]
            if schema == "uci" and feature == "studytime":
                values[feature] = st.select_slider(spec["labels"][i], options=[1, 2, 3, 4],
                    value=2, format_func=lambda x: STUDY_BANDS[x], key=f"{schema}_{feature}")
            else:
                values[feature] = st.slider(spec["labels"][i], lo, hi, spec["defaults"][i],
                                             key=f"{schema}_{feature}")
        threshold = st.slider("Probability threshold for “Likely pass”", .05, .95, .50, .05,
                              key=f"{schema}_threshold")
        st.caption("This changes the classification boundary, not the student's probability or the actual pass mark.")
    row = pd.DataFrame([values])
    p = float(pass_probability(bundle["model"], row)[0])
    st.session_state["last_pass_probability"] = p
    st.session_state[f"last_row_{schema}"] = row
    with right:
        a, b = st.columns(2)
        a.metric("Pass probability", f"{p:.1%}")
        b.metric("Fail probability", f"{1-p:.1%}")
        st.subheader(classification(p, threshold))
        st.progress(p, text=f"Estimated chance of passing: {p:.1%}")
        st.caption(f"{bundle['model_name']} · {bundle['metadata']['target_definition']}")
        st.write("This is a model estimate, not a guaranteed result or a measure of intelligence.")
        if abs(p - threshold) <= .1:
            st.info("This result is near your decision threshold. Small model differences could change the classification.")
    st.subheader("What happens when one factor changes?")
    feature = st.selectbox("Factor to explore", spec["features"],
                           format_func=lambda f: spec["labels"][spec["features"].index(f)])
    curve = what_if(bundle["model"], row, schema, feature)
    fig = px.line(curve, x=curve.columns[0], y="Pass probability", line_shape="hv",
                  color_discrete_sequence=["#2563eb"])
    fig.add_hline(y=threshold, line_dash="dash", line_color="#718096")
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    if feature == "studytime":
        fig.update_xaxes(tickmode="array", tickvals=list(STUDY_BANDS), ticktext=list(STUDY_BANDS.values()))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Other inputs are held fixed. The model uses intervals, so values within one interval have the same prediction. "
               "These curves show model sensitivity; changing a factor does not guarantee a causal improvement.")
    with st.expander("Download this anonymous scenario"):
        result = {**values, "pass_probability": p, "fail_probability": 1-p,
                  "classification": classification(p, threshold), "threshold": threshold,
                  "actual_pass_mark": bundle["metadata"]["pass_mark"]}
        st.download_button("Download prediction CSV", pd.DataFrame([result]).to_csv(index=False),
                           "prediction.csv", "text/csv")


def explain_bayes(bundle):
    schema = bundle["schema"]
    spec = SCHEMAS[schema]
    st.header("How Bayes' theorem produces the prediction")
    row = st.session_state.get(f"last_row_{schema}", pd.DataFrame([dict(zip(spec["features"], spec["defaults"]))]))
    st.caption("Using the latest profile from Prediction, or the default profile before your first edit.")
    st.dataframe(row.rename(columns=dict(zip(spec["features"], spec["labels"]))), hide_index=True, use_container_width=True)
    details = bayes_breakdown(bundle["raw_model"], row, schema)
    st.latex(r"P(Pass\mid x)=\frac{P(Pass)\prod_j P(b_j\mid Pass)}{\sum_{c\in\{Fail,Pass\}}P(c)\prod_jP(b_j\mid c)}")
    st.write("1. Start with the training class proportions: these are the prior probabilities.")
    a, b = st.columns(2)
    a.metric("Prior P(Pass)", f"{details['priors'][1]:.2%}")
    b.metric("Prior P(Fail)", f"{details['priors'][0]:.2%}")
    st.write("2. Put each input into its interval. Count how often that interval occurs within each outcome class.")
    st.dataframe(details["terms"].style.format({"P(interval | Fail)": "{:.4f}", "P(interval | Pass)": "{:.4f}"}),
                 hide_index=True, use_container_width=True)
    st.latex(r"P(b_j\mid c)=\frac{N_{j,b,c}+1}{N_c+4}")
    st.caption("Laplace smoothing adds one to each of four possible intervals so unseen intervals do not force a zero probability.")
    st.write("3. Multiply each class prior by its four likelihoods, then normalize.")
    jf, jp = details["joint"]
    st.code(f"Fail joint score = {jf:.8f}\nPass joint score = {jp:.8f}\n"
            f"P(Pass | inputs) = {jp:.8f} / ({jp:.8f} + {jf:.8f}) = {details['posterior'][1]:.4%}")
    a, b = st.columns(2)
    a.metric("Raw Bayes pass probability", f"{details['posterior'][1]:.2%}")
    final = float(pass_probability(bundle["model"], row)[0])
    b.metric("Dashboard pass probability", f"{final:.2%}")
    st.write("If calibrated Bayes was selected, a sigmoid fitted on training-fold predictions adjusts the raw probability. "
             "The raw Bayes calculation above is kept separate from that adjustment.")
    st.caption("Naive Bayes assumes the factors are independent given the outcome. Academic factors can be correlated, "
               "so this is an approximation, and calibration cannot guarantee reliable probabilities for every student.")


def statistics(bundle):
    stats = bundle["statistics"]
    schema = bundle["schema"]
    st.header("Patterns in student performance")
    st.caption("Training records only. These describe associations, not causes.")
    st.subheader("Mean and standard deviation")
    st.dataframe(stats["summary"].round(3), use_container_width=True)
    st.write("The mean is the average value. Sample standard deviation describes how spread out the values are. "
             "Count shows how many non-missing observations contributed.")
    if schema == "uci":
        st.info("studytime is an ordered code (1–4), not an exact number of hours. Its mean and standard deviation describe codes only.")
    with st.expander("Compare students who passed and failed"):
        st.dataframe(stats["by_outcome"].round(3), use_container_width=True)
    method = st.radio("Correlation method", ["Spearman (rank)", "Pearson (linear)"], horizontal=True)
    corr = stats["spearman" if method.startswith("Spearman") else "pearson"]
    fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1,
                    color_continuous_scale="RdBu_r", aspect="auto")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Correlation ranges from −1 to +1. pass = 1 and fail = 0. "
               "Spearman is useful for ordinal study-time bands. Constant columns have undefined correlation.")
    st.subheader("Observed conditional pass rates")
    feature = st.selectbox("Student factor", SCHEMAS[schema]["features"], key="stats_feature")
    observed = stats["conditional"].query("feature == @feature")
    fig = px.bar(observed, x="interval", y="observed_pass_rate", hover_data=["students", "passes"],
                 color_discrete_sequence=["#0d9488"])
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(observed, hide_index=True, use_container_width=True)
    st.caption("P(Pass | interval) = passes in that interval ÷ students in that interval. "
               "Empty intervals have no rate; small groups give unstable estimates. "
               "This is a single-factor observation, not the full model prediction.")


def performance(bundle):
    st.header("Model performance")
    st.caption(f"{len(bundle['train_indices'])} training records · {len(bundle['test_indices'])} held-out test records · seed 42")
    result = bundle["test_metrics"].iloc[0]
    for col, (key, label) in zip(st.columns(4), [
        ("accuracy", "Accuracy"), ("fail_recall", "Fail recall"),
        ("roc_auc", "ROC-AUC"), ("brier_score", "Brier score")]):
        col.metric(label, f"{result[key]:.3f}")
    st.write(f"Selected model: **{bundle['model_name']}**. Selection uses the lowest mean cross-validation "
             "Brier score on training data. Test results are evaluated afterwards.")
    st.dataframe(bundle["comparison"].round(4), hide_index=True, use_container_width=True)
    st.dataframe(bundle["test_metrics"].round(4), hide_index=True, use_container_width=True)
    st.caption("Lower Brier score and log loss mean better probability predictions. "
               "Fail recall measures the proportion of actual failing students identified. "
               "ROC-AUC measures ranking quality; accuracy alone does not measure probability reliability.")
    left, right = st.columns(2)
    with left:
        matrix = confusion_matrix(bundle["test_y"], bundle["test_probability"] >= .5, labels=[0, 1])
        fig = px.imshow(matrix, x=["Fail", "Pass"], y=["Fail", "Pass"], text_auto=True,
                        color_continuous_scale="Blues", labels={"x": "Predicted", "y": "Actual"},
                        title="Confusion matrix · threshold 0.50")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Ideal", line={"dash": "dash"})
        for label, p in [("Selected model", bundle["test_probability"]), ("Raw Bayes", bundle["raw_test_probability"])]:
            actual, predicted = calibration_curve(bundle["test_y"], p, n_bins=5, strategy="quantile")
            fig.add_scatter(x=predicted, y=actual, mode="lines+markers", name=label)
        fig.update_layout(title="Calibration · held-out records",
                          xaxis_title="Predicted pass probability", yaxis_title="Observed pass fraction")
        fig.update_xaxes(range=[0, 1])
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    fpr, tpr, _ = roc_curve(bundle["test_y"], bundle["test_probability"])
    fig = px.line(x=fpr, y=tpr, labels={"x": "False positive rate", "y": "True positive rate"},
                  title="ROC curve · Pass is the positive class")
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"dash": "dash"})
    st.plotly_chart(fig, use_container_width=True)
    st.caption("All test metrics use a fixed threshold of 0.50. The small test set limits precision; "
               "these scores do not establish accuracy for a different school or country.")


def methodology(bundle):
    st.header("Data and research method")
    meta = bundle["metadata"]
    st.write(meta["source"])
    st.success(meta["target_definition"])
    st.write(f"Records: **{meta['rows']}** · Pass: **{meta['pass_count']}** · Fail: **{meta['fail_count']}**")
    st.info(SCHEMAS[bundle["schema"]]["limitation"])
    st.markdown("""
1. Validate numeric inputs, remove exact duplicate source rows, and define pass/fail from final marks.
2. Reserve a stratified 20% test set; use the remaining 80% for training and exploration.
3. Convert inputs to fixed intervals and handle missing inputs inside each training fold.
4. Compare Naive Bayes with sigmoid-calibrated Naive Bayes using five-fold cross-validation.
5. Select by probability error (Brier score), fit on the training portion, then evaluate the held-out test set.
6. Display complementary pass/fail probabilities and allow interactive what-if exploration.
""")
    st.subheader("What the research can establish")
    st.write("The real-data model uses mathematics records from two Portuguese secondary schools. "
             "G1 and G2 are grades from the first two periods; the final grade G3 defines the outcome. "
             "Absence timing is not established by the dataset, so results are retrospective and may include later attendance information.")
    st.write("The four-factor CSV model needs actual historical attendance, study hours, previous marks, "
             "assignment scores and final outcomes. Never substitute generated values or rename exam marks as assignments.")
    st.subheader("Use and limitations")
    st.write("Predictions support learning and human review. They are not final academic judgments. "
             "Correlated factors challenge the Naive Bayes assumption; fixed intervals discard detail. "
             "A model trained on these schools needs local and future-cohort validation before institutional use. "
             "Gender is excluded as a predictor, but this does not guarantee fairness.")
    st.write("No names are requested. Custom records are processed in session memory, not written to files. "
             "Only upload anonymous records you are authorized to use.")
    st.markdown("[UCI dataset and license](https://doi.org/10.24432/C5TG7T) · "
                "[Naive Bayes reference](https://scikit-learn.org/stable/modules/naive_bayes.html) · "
                "[Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)")


def main():
    st.title(TITLE)
    st.caption("Understand the probability. Explore the factors. See the calculation.")
    st.sidebar.title("Student Performance")
    page = st.sidebar.radio("Page", ["Prediction", "Bayes explained", "Statistics", "Model performance", "Data and method"])
    try:
        bundle = choose_data()
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.stop()
    if bundle is None:
        st.stop()
    st.sidebar.caption(f"Model: {bundle['model_name']}")
    st.sidebar.caption(bundle["metadata"]["target_definition"])
    {"Prediction": prediction, "Bayes explained": explain_bayes, "Statistics": statistics,
     "Model performance": performance, "Data and method": methodology}[page](bundle)


if __name__ == "__main__":
    main()
