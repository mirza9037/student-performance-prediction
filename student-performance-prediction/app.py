"""Interactive pass/fail probability dashboard."""
from __future__ import annotations

import io
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, roc_curve
from src.config import MODEL_PATH, SCHEMAS, STUDY_BANDS
from src.modeling import bayes_breakdown, classification, pass_probability, what_if
from src.reporting import evidence_archive
from src.uncertainty import performance_intervals, wilson_interval
from src.ui import (
    PLOT_CONFIG,
    callout,
    context_bar,
    inject_theme,
    metric_strip,
    page_header,
    probability_result,
    reset_scroll_when_page_changes,
    section_header,
    sidebar_brand,
    sidebar_model_card,
    style_figure,
    workflow_strip,
)

st.set_page_config(page_title="StudPerf · Probability Atlas", page_icon="↗", layout="wide")
inject_theme()


def load_builtin():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model missing. Run python train.py, then restart the app.")
    # File bytes participate in Streamlit's cache key, so a cloud redeploy cannot
    # keep serving an older bundle from the previous no-argument cached loader.
    return load_model_bytes(MODEL_PATH.read_bytes())


@st.cache_resource
def load_model_bytes(payload: bytes):
    bundle = joblib.load(io.BytesIO(payload))
    if bundle.get("schema_version") != 2:
        raise ValueError("The saved model belongs to the old project. Run python train.py.")
    return ensure_research_evidence(bundle)


def ensure_research_evidence(bundle):
    """Upgrade older saved/session results using existing predictions, without refitting."""
    if "intervals" not in bundle:
        bundle["intervals"] = performance_intervals(
            bundle["test_y"], bundle["test_probability"], bundle["raw_test_probability"])
    conditional = bundle["statistics"]["conditional"]
    if not {"lower_95", "upper_95"}.issubset(conditional.columns):
        bounds = [wilson_interval(int(row.passes), int(row.students))
                  for row in conditional.itertuples()]
        conditional = conditional.copy()
        conditional[["lower_95", "upper_95"]] = bounds
        bundle["statistics"]["conditional"] = conditional
    return bundle


def open_probability_explanation():
    """Move the keyed navigation widget before Streamlit renders the next run."""
    st.session_state["page"] = "Understand the probability"


def prediction(bundle):
    schema = bundle["schema"]
    spec = SCHEMAS[schema]
    page_header("Student pass probability",
                "Set four study-related values and receive an immediate, explainable estimate.")
    dataset_name = bundle["metadata"].get("dataset_name", spec["name"])
    context_bar(("Dataset", dataset_name), ("Model", bundle["model_name"]),
                ("Target", bundle["metadata"]["target_definition"]))
    workflow_strip()
    left, right = st.columns([1, 1.15], gap="large")
    values = {}
    with left:
        section_header("Student profile", "No name or student ID is requested.")
        for i, feature in enumerate(spec["features"]):
            lo, hi = spec["bounds"][i]
            if schema == "uci" and feature == "studytime":
                values[feature] = st.select_slider(spec["labels"][i], options=[1, 2, 3, 4],
                    value=2, format_func=lambda x: STUDY_BANDS[x], key=f"{schema}_{feature}")
            else:
                values[feature] = st.slider(spec["labels"][i], lo, hi, spec["defaults"][i],
                                             key=f"{schema}_{feature}")
        with st.expander("Advanced decision setting"):
            threshold = st.slider("Probability threshold for “Likely pass”", .05, .95, .50, .05,
                                  key=f"{schema}_threshold")
            st.caption("This changes only the classification label. It does not change the probability or actual pass mark.")
    row = pd.DataFrame([values])
    p = float(pass_probability(bundle["model"], row)[0])
    st.session_state["last_pass_probability"] = p
    st.session_state[f"last_row_{schema}"] = row
    with right:
        section_header("Probability reading", "A model estimate, never a guaranteed result.")
        probability_result(p, threshold, bundle["model_name"], bundle["metadata"]["target_definition"])
        if abs(p - threshold) <= .1:
            callout("Near the decision boundary",
                    "Small model differences could change the classification. Review the evidence before acting.",
                    "amber")
        st.button(
            "Understand this result",
            width="stretch",
            on_click=open_probability_explanation,
        )
    section_header("Sensitivity map", "Change one factor while the other inputs stay fixed.")
    feature = st.selectbox("Factor to explore", spec["features"],
                           format_func=lambda f: spec["labels"][spec["features"].index(f)])
    curve = what_if(bundle["model"], row, schema, feature)
    fig = px.line(curve, x=curve.columns[0], y="Pass probability", line_shape="hv",
                  color_discrete_sequence=["#2563eb"])
    fig.add_hline(y=threshold, line_dash="dash", line_color="#718096")
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    if feature == "studytime":
        fig.update_xaxes(tickmode="array", tickvals=list(STUDY_BANDS), ticktext=list(STUDY_BANDS.values()))
    st.plotly_chart(style_figure(fig, "cobalt"), config=PLOT_CONFIG)
    st.caption("Other inputs are held fixed. The model uses intervals, so values within one interval have the same prediction. "
               "These curves show model sensitivity; changing a factor does not guarantee a causal improvement.")
    with st.expander("Dataset boundary and responsible use"):
        callout("Interpret with context", spec["limitation"], "amber")
        st.write("This estimate supports learning and human review. It is not a guaranteed result, a measure of intelligence, "
                 "or a substitute for local validation and academic judgment.")
    with st.expander("Download this anonymous scenario"):
        result = {**values, "pass_probability": p, "fail_probability": 1-p,
                  "classification": classification(p, threshold), "threshold": threshold,
                  "actual_pass_mark": bundle["metadata"]["pass_mark"]}
        st.download_button("Download prediction CSV", pd.DataFrame([result]).to_csv(index=False),
                           "prediction.csv", "text/csv")


def explain_bayes(bundle):
    schema = bundle["schema"]
    spec = SCHEMAS[schema]
    page_header("How this probability was calculated",
                "Follow the latest student profile from the starting odds to the final estimate.")
    context_bar(("Method", "Conditional probability"), ("Rule", "Bayes' theorem"),
                ("Assumption", "Conditional independence"))
    row = st.session_state.get(f"last_row_{schema}", pd.DataFrame([dict(zip(spec["features"], spec["defaults"]))]))
    st.caption("Using the latest profile from Predict, or the default profile before your first edit.")
    st.dataframe(row.rename(columns=dict(zip(spec["features"], spec["labels"]))), hide_index=True, width="stretch")
    details = bayes_breakdown(bundle["raw_model"], row, schema)
    section_header("The equation", "Bayes combines a prior with the evidence from each interval.")
    st.latex(r"P(Pass\mid x)=\frac{P(Pass)\prod_j P(b_j\mid Pass)}{\sum_{c\in\{Fail,Pass\}}P(c)\prod_jP(b_j\mid c)}")
    st.write("1. Start with the training class proportions: these are the prior probabilities.")
    metric_strip([
        ("Prior P(Pass)", f"{details['priors'][1]:.2%}", "Training class proportion"),
        ("Prior P(Fail)", f"{details['priors'][0]:.2%}", "Training class proportion"),
    ])
    st.write("2. Put each input into its interval. Count how often that interval occurs within each outcome class.")
    st.dataframe(details["terms"].style.format({"P(interval | Fail)": "{:.4f}", "P(interval | Pass)": "{:.4f}"}),
                 hide_index=True, width="stretch")
    st.latex(r"P(b_j\mid c)=\frac{N_{j,b,c}+1}{N_c+4}")
    st.caption("Laplace smoothing adds one to each of four possible intervals so unseen intervals do not force a zero probability.")
    st.write("3. Multiply each class prior by its four likelihoods, then normalize.")
    jf, jp = details["joint"]
    st.code(f"Fail joint score = {jf:.8f}\nPass joint score = {jp:.8f}\n"
            f"P(Pass | inputs) = {jp:.8f} / ({jp:.8f} + {jf:.8f}) = {details['posterior'][1]:.4%}")
    final = float(pass_probability(bundle["model"], row)[0])
    metric_strip([
        ("Raw Bayes", f"{details['posterior'][1]:.2%}", "Before calibration"),
        ("Dashboard result", f"{final:.2%}", "After model selection"),
    ])
    st.write("If calibrated Bayes was selected, a sigmoid fitted on training-fold predictions adjusts the raw probability. "
             "The raw Bayes calculation above is kept separate from that adjustment.")
    st.caption("Naive Bayes assumes the factors are independent given the outcome. Academic factors can be correlated, "
               "so this is an approximation, and calibration cannot guarantee reliable probabilities for every student.")


def statistics(bundle):
    stats = bundle["statistics"]
    schema = bundle["schema"]
    page_header("What the student data shows",
                "Explore averages, variation, relationships, and observed pass rates without claiming causation.")
    context_bar(("Sample", f"{len(bundle['train_indices'])} training records"),
                ("Scope", "Descriptive"), ("Outcome", "Pass / fail"))
    section_header("Center and spread", "Mean, sample standard deviation, and observation count.")
    st.dataframe(stats["summary"].round(3), width="stretch")
    st.write("The mean is the average value. Sample standard deviation describes how spread out the values are. "
             "Count shows how many non-missing observations contributed.")
    with st.expander("Why these statistical measures are used"):
        st.latex(r"\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i,\qquad s=\sqrt{\frac{\sum_{i=1}^{n}(x_i-\bar{x})^2}{n-1}}")
        st.write("Mean and sample standard deviation summarize the center and spread of observed inputs. "
                 "Pearson correlation measures linear association; Spearman compares ranks and is suitable "
                 "for ordered study-time bands. Neither establishes that a factor causes passing.")
        st.write("Missing observations are excluded from summaries and pairwise correlations. "
                 "The model handles missing inputs separately inside each training fold.")
    if schema == "uci":
        callout("Ordered study-time codes",
                "Study time is recorded from 1–4, not as exact hours. Its mean and standard deviation describe codes only.",
                "amber")
    with st.expander("Compare students who passed and failed"):
        st.dataframe(stats["by_outcome"].round(3), width="stretch")
    section_header("Association map", "Switch between rank and linear correlation.")
    method = st.radio("Correlation method", ["Spearman (rank)", "Pearson (linear)"], horizontal=True)
    corr = stats["spearman" if method.startswith("Spearman") else "pearson"]
    fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1,
                    color_continuous_scale="RdBu_r", aspect="auto")
    st.plotly_chart(style_figure(fig, "cobalt"), config=PLOT_CONFIG)
    st.caption("Correlation ranges from −1 to +1. pass = 1 and fail = 0. "
               "Spearman is useful for ordinal study-time bands. Constant columns have undefined correlation.")
    section_header("Observed conditional pass rates", "A single-factor view with 95% Wilson intervals.")
    feature = st.selectbox("Student factor", SCHEMAS[schema]["features"], key="stats_feature")
    observed = stats["conditional"].query("feature == @feature").copy()
    observed["error_plus"] = (observed.upper_95 - observed.observed_pass_rate).clip(lower=0)
    observed["error_minus"] = (observed.observed_pass_rate - observed.lower_95).clip(lower=0)
    fig = px.bar(observed, x="interval", y="observed_pass_rate", hover_data=["students", "passes"],
                 error_y="error_plus", error_y_minus="error_minus",
                 color_discrete_sequence=["#0d9488"])
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    st.plotly_chart(style_figure(fig, "teal"), config=PLOT_CONFIG)
    st.dataframe(observed.drop(columns=["error_plus", "error_minus"]), hide_index=True, width="stretch")
    st.caption("P(Pass | interval) = passes in that interval ÷ students in that interval. "
               "Empty intervals have no rate; small groups give unstable estimates. "
               "This is a single-factor observation, not the full model prediction.")
    st.caption("Error bars are approximate 95% Wilson confidence intervals for observed pass rates. "
               "They describe group proportions, not an individual student's chance. "
               "These exploratory intervals are not adjusted for multiple comparisons.")


def performance(bundle):
    page_header("How reliable is the model?",
                "Review classification and probability quality on records the model did not train on.")
    context_bar(("Training", f"{len(bundle['train_indices'])} records"),
                ("Held-out test", f"{len(bundle['test_indices'])} records"), ("Seed", "42"))
    result = bundle["test_metrics"].iloc[0]
    metric_strip([
        ("Accuracy", f"{result['accuracy']:.3f}", "Overall decisions"),
        ("Fail recall", f"{result['fail_recall']:.3f}", "At-risk students found"),
        ("ROC–AUC", f"{result['roc_auc']:.3f}", "Ranking quality"),
        ("Brier score", f"{result['brier_score']:.3f}", "Lower is better"),
    ], columns=4)
    section_header("Selection evidence", "Cross-validation chooses the model before the test set is opened.")
    st.write(f"Selected model: **{bundle['model_name']}**. Selection uses the lowest mean cross-validation "
             "Brier score on training data. Test results are evaluated afterwards.")
    with st.expander("Cross-validation comparison", expanded=True):
        st.dataframe(bundle["comparison"].round(4), hide_index=True, width="stretch")
    with st.expander("Held-out test metrics"):
        st.dataframe(bundle["test_metrics"].round(4), hide_index=True, width="stretch")
    st.caption("Lower Brier score and log loss mean better probability predictions. "
               "Fail recall measures the proportion of actual failing students identified. "
               "ROC-AUC measures ranking quality; accuracy alone does not measure probability reliability.")
    section_header("Uncertainty", "Intervals show what this finite test sample can and cannot establish.")
    intervals = bundle["intervals"]
    st.dataframe(intervals.round(4), hide_index=True, width="stretch")
    st.write(calibration_finding(bundle))
    st.caption("Accuracy and recall: 95% Wilson intervals. Brier score and its paired model difference: "
               "2,000 class-stratified bootstrap resamples, seed 42. The selected-minus-raw difference "
               "is negative when the selected model has lower error. These intervals hold the fitted "
               "models and observed class counts fixed; they exclude training variability, school clustering "
               "and population change. They are not confidence intervals for an individual prediction.")
    st.download_button("Download confidence intervals", intervals.to_csv(index=False),
                       "confidence_intervals.csv", "text/csv")
    left, right = st.columns(2)
    with left:
        matrix = confusion_matrix(bundle["test_y"], bundle["test_probability"] >= .5, labels=[0, 1])
        fig = px.imshow(matrix, x=["Fail", "Pass"], y=["Fail", "Pass"], text_auto=True,
                        color_continuous_scale="Blues", labels={"x": "Predicted", "y": "Actual"},
                        title="Confusion matrix · threshold 0.50")
        st.plotly_chart(style_figure(fig, "cobalt"), config=PLOT_CONFIG)
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
        st.plotly_chart(style_figure(fig, "teal"), config=PLOT_CONFIG)
    fpr, tpr, _ = roc_curve(bundle["test_y"], bundle["test_probability"])
    fig = px.line(x=fpr, y=tpr, labels={"x": "False positive rate", "y": "True positive rate"},
                  title="ROC curve · Pass is the positive class")
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"dash": "dash"})
    st.plotly_chart(style_figure(fig, "cobalt"), config=PLOT_CONFIG)
    st.caption("All test metrics use a fixed threshold of 0.50. The small test set limits precision; "
               "these scores do not establish accuracy for a different school or country.")


def methodology(bundle):
    page_header("How the study was run",
                "See where the evidence comes from, how the test stays fair, and where the conclusions stop.")
    meta = bundle["metadata"]
    context_bar(("Records", str(meta["rows"])), ("Pass", str(meta["pass_count"])),
                ("Fail", str(meta["fail_count"])), ("Split", "80 / 20"))
    section_header("Source and target")
    st.write(meta["source"])
    callout("Target definition", meta["target_definition"], "teal")
    callout("Dataset boundary", SCHEMAS[bundle["schema"]]["limitation"], "amber")
    section_header("Experimental protocol", "A leakage-resistant path from validated rows to held-out evidence.")
    st.markdown("""
1. Validate numeric inputs, remove exact duplicate source rows, and define pass/fail from final marks.
2. Reserve a stratified 20% test set; use the remaining 80% for training and exploration.
3. Convert inputs to fixed intervals and handle missing inputs inside each training fold.
4. Compare Naive Bayes with sigmoid-calibrated Naive Bayes using five-fold cross-validation.
5. Select by probability error (Brier score), fit on the training portion, then evaluate the held-out test set.
6. Display complementary pass/fail probabilities and allow interactive what-if exploration.
""")
    section_header("What the research can establish")
    if bundle["schema"] == "uci":
        course = bundle["metadata"].get("course", "Portuguese language")
        st.write(f"The real-data model uses {course.lower()} course records from two Portuguese secondary schools. "
                 "G1 and G2 are grades from the first two periods; the final grade G3 defines the outcome. "
                 "Absence timing is not established by the dataset, so results are retrospective and may include later attendance information.")
    else:
        st.write("These results use your uploaded four-factor records. Their source, measurement timing and "
                 "representativeness have not been independently verified. They are separate from the UCI experiment.")
    section_header("Use and limitations")
    st.write("Predictions support learning and human review. They are not final academic judgments. "
             "Correlated factors challenge the Naive Bayes assumption; fixed intervals discard detail. "
             "A model trained on these schools needs local and future-cohort validation before institutional use. "
             "Gender is excluded as a predictor, but this does not guarantee fairness.")
    st.write("No names or student identifiers are used. The public app loads only the committed aggregate model artifact; "
             "it does not accept or store student datasets.")
    st.markdown("[UCI dataset and license](https://doi.org/10.24432/C5TG7T) · "
                "[Original research paper (PDF)](https://qn1-next.xuetangonline.com/17744286543388.pdf) · "
                "[Naive Bayes reference](https://scikit-learn.org/stable/modules/naive_bayes.html) · "
                "[Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)")


def calibration_finding(bundle):
    """Describe the paired comparison without treating a point estimate as proof."""
    row = bundle["intervals"].set_index("metric").loc["brier_selected_minus_raw"]
    if bundle["model_name"] == "Naive Bayes":
        return "Raw Naive Bayes was selected by training cross-validation, so the selected-versus-raw comparison is identical (difference zero)."
    direction = "lower" if row.estimate < 0 else "higher" if row.estimate > 0 else "equal"
    conclusion = ("The interval includes zero, so this test sample does not establish a clear probability-error improvement."
                  if row.lower_95 <= 0 <= row.upper_95 else
                  "The interval lies below zero, supporting lower error for this fitted model on this test population."
                  if row.upper_95 < 0 else
                  "The interval lies above zero, indicating higher error on this test sample.")
    return (f"Selected Bayes has {direction} held-out Brier error than raw Bayes: difference "
            f"{row.estimate:+.4f}, 95% interval [{row.lower_95:+.4f}, {row.upper_95:+.4f}]. {conclusion}")


def research(bundle):
    page_header("Evidence for the research paper",
                "Follow the research question through the experiment, findings, uncertainty, and limitations.")
    spec = SCHEMAS[bundle["schema"]]
    meta = bundle["metadata"]
    context_bar(("Dataset", meta.get("dataset_name", spec["name"])), ("Model", bundle["model_name"]),
                ("Evidence", "Held-out test"))
    callout("Research question",
            "How effectively can Naive Bayes estimate pass/fail probabilities from prior academic performance and study-related factors, and does calibration improve those estimates?",
            "cobalt")
    section_header("What is actually measured?", "The proposal factors are mapped to the variables the dataset really contains.")
    if bundle["schema"] == "uci":
        st.dataframe(pd.DataFrame({
            "Proposal factor": ["Attendance", "Study hours", "Previous marks", "Assignment performance"],
            "Available research measurement": ["Number of absences; no attendance percentage",
                "Ordered weekly study-time bands; no exact hours", "G1 and G2, each on a 0–20 scale",
                "Not recorded in the UCI dataset"],
        }), hide_index=True, width="stretch")
        st.write("This experiment evaluates only the available UCI measurements. Assignment scores, attendance percentages, "
                 "and exact study hours were not studied and are not implied by these results.")
    else:
        st.dataframe(pd.DataFrame({"Column": spec["features"], "Measurement": spec["labels"]}),
                     hide_index=True, width="stretch")
        st.write("All four proposal factors are used. Final marks construct the outcome only and are never a predictor.")
    section_header("Experiment at a glance", "The held-out test never participates in model selection.")
    st.graphviz_chart(r'''digraph {
      rankdir=LR;
      node [shape=box, style=rounded];
      data [label="Validate data\nDefine pass/fail"];
      train [label="80% training\nStatistics + nested CV"];
      select [label="Select by CV Brier\nFit on training only"];
      test [label="20% held-out test"];
      evaluate [label="Metrics + uncertainty\nCompare with baseline"];
      data -> train -> select -> evaluate;
      data -> test -> evaluate;
    }''')
    st.caption("The test set never chooses the model. Raw and calibrated Bayes use the same split; "
               "a training-prior baseline shows how a model with no student-specific inputs performs.")
    section_header("Findings from this experiment")
    assoc = bundle["statistics"]["spearman"]["pass"].drop("pass").dropna()
    if not assoc.empty:
        strongest = assoc.abs().idxmax()
        label = spec["labels"][spec["features"].index(strongest)]
        st.write(f"The strongest training-set rank association with pass/fail among the selected inputs "
                 f"is {label}: Spearman correlation {assoc[strongest]:+.3f}. "
                 "This is descriptive, not a causal effect or an independent significance test.")
    results = bundle["test_metrics"].set_index("model")
    selected = results.loc["Selected Bayes model"]
    baseline = results.loc["Training-prior baseline"]
    st.write(f"On {len(bundle['test_indices'])} held-out records, selected-model accuracy is "
             f"{selected.accuracy:.1%}, compared with {baseline.accuracy:.1%} for the training-prior baseline. "
             f"Brier scores are {selected.brier_score:.4f} and {baseline.brier_score:.4f}, respectively; lower is better.")
    st.write(calibration_finding(bundle))
    section_header("Limits on the conclusion")
    st.write("These results describe one dataset and one split. They do not establish performance at another "
             "institution. Correlated inputs challenge the Bayes independence assumption; a probability "
             "estimate is not a guaranteed result. Future-cohort testing and local validation remain necessary.")
    if bundle["schema"] == "uci":
        callout("Retrospective evidence",
                "This demonstration uses the second-period grade. Absence timing is not established, so prospective early-warning accuracy is not proven.",
                "red")
    section_header("Download evidence for your paper", "Export aggregate results, method notes, and chart-ready data.")
    st.write("Download model comparisons, fold results, descriptive statistics, correlations, conditional "
             "rates, confidence intervals and method notes for the active dataset. Use each chart's "
             "download control to save a PNG figure.")
    st.download_button("Download research evidence (ZIP)", evidence_archive(bundle),
                       f"{bundle['schema']}_research_evidence.zip", "application/zip", key="evidence_zip")
    st.caption("The archive contains aggregate results only. It is supporting evidence, not a completed conference paper.")
    st.markdown("[Project code and instructions](https://github.com/mirza9037/student-performance-prediction) · "
                "[UCI source](https://doi.org/10.24432/C5TG7T) · "
                "[Cortez & Silva paper (PDF)](https://qn1-next.xuetangonline.com/17744286543388.pdf) · "
                "[Wilson intervals (NIST)](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm) · "
                "[Bootstrap methodology (SciPy)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)")


def main():
    sidebar_brand()
    page = st.sidebar.radio(
        "Page",
        ["Predict", "Understand the probability", "Explore the data", "Check model quality", "Read the method", "Research evidence"],
        key="page",
    )
    reset_scroll_when_page_changes(page)
    try:
        bundle = load_builtin()
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.stop()
    bundle = ensure_research_evidence(bundle)
    result = bundle["test_metrics"].iloc[0]
    sidebar_model_card(
        bundle["metadata"].get("dataset_name", SCHEMAS[bundle["schema"]]["name"]),
        bundle["metadata"]["rows"],
        result["accuracy"],
    )
    st.sidebar.caption(f"{bundle['model_name']} · Pass mark 10/20")
    {
        "Predict": prediction,
        "Understand the probability": explain_bayes,
        "Explore the data": statistics,
        "Check model quality": performance,
        "Read the method": methodology,
        "Research evidence": research,
    }[page](bundle)


if __name__ == "__main__":
    main()
