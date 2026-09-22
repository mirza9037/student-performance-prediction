# AI-Based Student Performance Prediction Using Probability

A complete research prototype that estimates the **probability of recorded student dropout**, rather than only a Pass/Fail label. It compares information available at enrollment with information available after the first semester. The dashboard is a decision-support demonstration, not a validated institutional deployment.

## Main features

- Two prediction stages with strict feature allowlists; no Gender or second-semester prediction inputs.
- Logistic Regression, Gaussian Naive Bayes, Random Forest, and HistGradientBoosting.
- Five-fold training cross-validation with separate five-fold sigmoid calibration inside each fold.
- One shared, untouched 20% test set for final evaluation of selected models.
- Probability metrics, calibration diagrams, fairness diagnostics, bootstrap intervals and permutation importance.
- Four Streamlit pages: Individual Prediction, Model Performance, Research Insights, Ethics and Limitations.
- Reproducible seed 42, source checksum, generated report, Windows launchers and executed tests.

## Project structure

```text
student-performance-prediction/
├── app.py
├── train.py
├── requirements.txt
├── requirements-lock.txt             # exact environment from the verified run
├── README.md
├── setup_windows.bat
├── run_app.bat
├── .gitignore
├── data/raw/uci_student_dropout.csv
├── models/model_bundle.joblib
├── reports/
│   ├── research_report.md
│   ├── model_comparison.csv           # training CV only; all eight candidates
│   ├── cv_fold_metrics.csv
│   ├── test_metrics.csv               # only the selected model per stage
│   ├── test_confidence_intervals.csv
│   ├── calibration_bins.csv
│   ├── fairness_audit.csv
│   ├── feature_importance.csv
│   ├── experiment_metadata.json
│   └── figures/
│       ├── confusion_matrix.png       # first-semester convenience copies
│       ├── calibration_curve.png
│       ├── roc_curve.png
│       ├── feature_importance.png
│       ├── enrollment/               # all four stage-specific figures
│       └── first_semester/
├── scripts/smoke_server.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── modeling.py
│   ├── evaluation.py
│   └── reporting.py
└── tests/test_project.py
```

## Dataset source

Official [UCI Predict Students’ Dropout and Academic Success](https://doi.org/10.24432/C5MC89), CC BY 4.0. The data describe Portuguese higher education. The loader tries the supplied [CSV endpoint](https://archive.ics.uci.edu/static/public/697/data.csv), then the official UCI ZIP archive if necessary. Both are official sources. It validates the schema before saving. No synthetic data fallback is used: download failures produce an actionable error.

Target: **Dropout = 1; Graduate or Enrolled = 0**. The latter combines completed and unresolved outcomes, so the score is not a general academic-failure probability. The downloaded CSV is cached and reused. Delete only that project-owned CSV if you intentionally want a fresh download.

## Installation on Windows

Install Python **3.11 or newer**, including pip, and enable “Add Python to PATH.” Open this project folder in a terminal.

The simplest setup is:

```powershell
.\setup_windows.bat
```

This creates `.venv`, activates it, upgrades pip, installs dependencies and trains both stages. Training fits many nested models, so allow several minutes after downloads finish.

Streamlit is pinned to 1.50.0: newer wheels include deeply nested bundled assets that can exceed Windows' default path limit when installed in long project folders. This project does not require changing Windows registry settings.

Manual setup without changing PowerShell's execution policy:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py
```

For the exact environment used in the delivered run, replace `requirements.txt` with `requirements-lock.txt`. The lock records a Windows/Python 3.12 environment; other platforms may need the bounded main requirements. Models should be retrained after changing scikit-learn versions. Load joblib artifacts only from trusted sources.

On macOS/Linux, the equivalent interpreter is `.venv/bin/python` after `python3 -m venv .venv`.

## Train or regenerate the research outputs

```powershell
.\.venv\Scripts\python.exe train.py
```

Or, after activating the environment, `python train.py`. The command downloads and validates the official data if missing, prints target distribution and fold progress, selects models, fits calibrated final models, and writes reports, figures and the bundle. It replaces only this project's generated output files; it never refits in response to dashboard entries. Relative paths are resolved from the project itself.

The two model selections are frozen before either is tested. The saved model bundle contains both calibrated models, feature names, training medians/ranges for the form, metrics and provenance. It contains no submitted dashboard profiles.

## Launch the dashboard

```powershell
.\run_app.bat
```

Or:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

Open the local URL printed by Streamlit (normally `http://127.0.0.1:8501`). Press **Ctrl+C** in the terminal to stop. Choose a prediction stage in the sidebar, enter values using the source grading scales, and click **Calculate risk probability**. Initial form values are training medians, not an actual student. Recalculate after editing a profile.

Low risk is below 30%, Medium is 30% to below 60%, and High is at least 60%. The threshold slider changes the support-review flag; it does not change the probability or these bands. Performance figures use the original 0.50 threshold. Any operational threshold must be validated by the institution.

The form collects no names or identifiers and does not save submissions to disk. Enter only anonymous information. Dataset scales and economic values cannot be assumed equivalent to Pakistani values.

## Testing and execution checks

Run these after training; integration tests deliberately require the real generated artifacts:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe -m compileall -q app.py train.py src tests scripts
.\.venv\Scripts\python.exe scripts\smoke_server.py
```

Tests validate input rejection, deduplication, split isolation, feature exclusions, fitted train-only medians, finite probabilities, saved artifact consistency, group denominators, threshold behavior, and actual Streamlit rendering/submission in both modes using AppTest. The smoke script launches a local server on an available port, checks its health and HTML, and always terminates its own process. Server health alone does not execute the app: AppTest supplies the complementary page-execution check. Results are recorded in `reports/verification.txt` and `reports/server_smoke.json` in the delivered project.

## Research methodology

Exact duplicate rows are removed before a seed-42 stratified 80/20 split. Median imputation and optional scaling are inside each classifier pipeline. Both stages use the same train/test membership. Five outer folds compare each sigmoid-calibrated model; five inner folds supply out-of-fold scores for calibration. Inner preprocessing cannot see outer validation rows. The final calibrated model is fitted using training data only.

Model selection prioritizes mean CV **PR-AUC, implemented as average precision**, then recall, then lower Brier score, then F1. This is a lexicographic ranking with secondary criteria as tie-breakers. Accuracy is secondary. Class weighting is enabled for Logistic Regression, Random Forest and HistGradientBoosting; GaussianNB uses empirical priors and has no built-in class_weight argument.

The test set evaluates only the selected model per stage. It also supports the fixed-threshold Gender audit, ten-repeat permutation importance, and descriptive bootstrap intervals. None of these diagnostic results feeds back into model selection. Detailed methods and actual results are in [the research report](reports/research_report.md).

## Limitations and responsible use

This is a retrospective single-source experiment. Unknown collection timestamps, unresolved Enrolled outcomes and the random rather than temporal split limit early-warning claims. Debtor, tuition and scholarship status must genuinely be measured at the selected prediction time. Local institutions must verify that assumption or remove the affected fields and retrain.

Calibration is an estimated correction, not a guarantee. Portuguese-to-Pakistani transfer requires local cohort validation, scale alignment, fairness review and likely retraining/recalibration. Gender exclusion does not remove proxies. Group differences need investigation and do not automatically prove discrimination. Feature importance is not causal evidence. Human review is required before every intervention, and scores must never be treated as measures of intelligence or potential.

## Screenshots

Placeholder for your final-year submission: add screenshots of Individual Prediction, Model Performance, Research Insights and Ethics and Limitations after launching the dashboard. No student identifiers should appear. Generated evaluation figures are already available in `reports/figures/`.

## References

- [UCI dataset](https://doi.org/10.24432/C5MC89)
- [Scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [UNESCO Recommendation on the Ethics of Artificial Intelligence](https://www.unesco.org/en/articles/recommendation-ethics-artificial-intelligence)
