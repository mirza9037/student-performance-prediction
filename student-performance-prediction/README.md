# AI-Based Student Performance Prediction Using Probability

**Live app: [studperf.streamlit.app](https://studperf.streamlit.app/)**. The research evidence page provides downloadable results.

A Python research application that applies **Bayes' theorem** to estimate a student's probability of passing or failing.

## What the app does

1. **Prediction:** adjust student factors to see pass/fail probabilities, a likely-pass/fail classification and a what-if chart.
2. **Bayes explained:** inspect training priors, each conditional likelihood, the normalization calculation and any calibration adjustment.
3. **Statistics:** inspect means, sample standard deviations, outcome-group summaries, Pearson/Spearman correlations and observed conditional pass rates.
4. **Model performance:** compare raw and calibrated Bayes, then inspect held-out metrics, confusion matrix, ROC and calibration.
5. **Data and method:** review target definitions, data provenance, experimental design and limitations.
6. **Research evidence:** see the research question, actual measurements, experiment diagram, findings and a ZIP download for the active dataset.

The Statistics page shows 95% Wilson error bars on observed conditional pass rates.
The Model performance page includes Wilson intervals for accuracy and class recall, plus 2,000 paired,
class-stratified percentile bootstrap resamples for Brier error and the selected-minus-raw Brier difference.
Negative differences favor the selected model. These intervals describe aggregate test-sample uncertainty,
not individual prediction confidence; they exclude training/selection variability, school clustering and population drift.

The research ZIP contains CSV summaries, CV/test metrics, confidence intervals, experiment metadata and method notes.
It excludes student rows and model binaries.
Download chart PNGs with the Plotly controls. The notes support the paper; they are not a finished conference submission.

## Data: one verified research cohort

The app uses only the official Portuguese-language cohort from [UCI Student Performance](https://doi.org/10.24432/C5TG7T), collected from two Portuguese secondary schools and provided under CC BY 4.0. It has 649 records (549 passes, 100 fails) under the declared **G3 ≥10/20** pass definition.

Its actual predictors are:
- `absences`: number of school absences (not attendance percentage).
- `studytime`: weekly study-time band, 1: <2h, 2: 2–5h, 3: 5–10h, 4: >10h.
- `G1`: first-period grade, 0–20.
- `G2`: second-period grade, 0–20.

G3 (final grade) defines the target and is never a predictor. G1/G2 are not assignment scores.
Absence collection timing is unspecified; this is a retrospective demonstration after G2, not a validated early-warning forecast.

No assignment scores, attendance percentages or exact study hours are fabricated to fill the UCI gaps. The public website does not accept replacement datasets or retrain the model.

## Installation and launch

Use **Python 3.12** for compatibility with the committed model. On Windows:

```powershell
cd student-performance-prediction
.\setup_windows.bat
.\run_app.bat
```

Or manually:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The included trained model permits immediate launch once dependencies are installed.
Run Streamlit from the repository root with `streamlit_app.py` for cloud parity.

## Training and reproducibility

```powershell
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/smoke_server.py
```

The official dataset is downloaded only when missing. Source URL, license and SHA-256 are recorded beside it.
There is no synthetic fallback.

## Method

Inputs are assigned to four predeclared intervals. Categorical Naive Bayes estimates:

`P(Pass | x) = P(Pass) × product P(input_interval | Pass) / sum of both class joint scores`

Class priors come from training frequencies. Conditional likelihoods use Laplace smoothing:
`(class-and-interval count + 1) / (class count + 4)`.
Fail probability is `1 − pass probability`. The app verifies the raw formula against the fitted model.

Categorical bins are appropriate for the study-time bands and make conditional probabilities easy to explain.
Within-bin edits do not change probability. No monotonic or causal improvement is promised.
Naive Bayes assumes conditional independence, which correlated grades can violate.

A reproducible stratified 80/20 split (seed 42) reserves test data.
Raw Bayes and sigmoid-calibrated Bayes are compared on five training folds.
Calibration has five inner folds; preprocessing is fitted inside each fold.
The version with the lowest mean Brier score is selected (log loss breaks ties), then evaluated on the held-out test set.
The saved models remain trained on only the 80% training portion.
The dashboard distinguishes raw Bayes from the calibrated output.

Statistics use training records only:
- Mean and sample standard deviation (ddof=1), excluding missing values.
- Pearson correlation for linear association; Spearman for ranked association.
- Study-time code summaries are **not hours**.
- Observed P(Pass | interval) includes counts; it differs from the model's P(interval | class).

The actual grade pass mark and prediction decision threshold are different. The latter defaults to probability 0.50 and affects only classification. The reported test metrics always use 0.50.

## Results and files

Single Portuguese model: calibrated categorical Naive Bayes. On 130 held-out records: accuracy 0.9077, pass precision 0.9537, pass recall 0.9364, fail recall 0.7500, pass F1 0.9450, ROC-AUC 0.9511, pass average precision 0.9905, Brier 0.0703 and log loss 0.2369.

These results apply only to this fixed UCI cohort.

- `app.py`: six focused research and prediction views powered by one model.
- `train.py`: reproducible offline experiment.
- `src/config.py`: schemas, ranges and fixed bins.
- `src/data.py`: download, validation and target construction.
- `src/modeling.py`: Bayes pipeline, explanation and what-if calculations.
- `src/evaluation.py`: nested CV, selection and metrics.
- `src/uncertainty.py`: Wilson intervals and reproducible paired bootstrap analysis.
- `src/reporting.py`: statistics, figures and generated research report.
- `models/model_bundle.joblib`: trusted trained artifact (never load untrusted joblib files).
- `reports/`: complete metrics, descriptive CSVs, figures, metadata and report.
- `tests/test_project.py`: data validation, exact Bayes math, leakage checks and UI tests.

## Deployment

Use Streamlit Community Cloud with repository `mirza9037/student-performance-prediction`,
branch `master`, entrypoint `streamlit_app.py`, Python **3.12**, no secrets.
Root runtime pins match the saved model. Enable public sharing after deployment.
If the repository is private, grant Streamlit access through its official GitHub connection flow.

## Limitations

The small historical dataset cannot establish performance at a new institution or in Pakistan.
G1/G2 are related to the final grade; absence timing is unresolved.
Correlations and what-if charts are not causal claims. Calibration quality varies by cohort.
No subgroup fairness conclusion is claimed; excluding gender does not remove proxy bias.
Use human review and supportive interventions, never treat a prediction as a guaranteed result.
Repeated students require grouped validation; this prototype assumes one record per student.

## Screenshots

The running app contains Prediction, Bayes explained, Statistics, Model performance and Data and method pages.
Scientific charts are saved under `reports/figures/`.

## References

- Cortez, P. (2008), [UCI Student Performance](https://doi.org/10.24432/C5TG7T), CC BY 4.0.
- [Scikit-learn Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html).
- [Scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html).
