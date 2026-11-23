# AI-Based Student Performance Prediction Using Probability

**Public website: [studperf.streamlit.app](https://studperf.streamlit.app/)** — open in a browser without installing Python or signing in.

Predict **pass and fail probabilities** with Naive Bayes, explore student factors, and see the calculation.

This revision replaces the earlier dropout system. It includes:
- Complementary pass/fail probabilities and a clear likely-pass/likely-fail classification.
- Interactive input changes and one-factor what-if charts.
- A worked Bayes calculation with priors, conditional likelihoods and posterior probabilities.
- Means, sample standard deviations, Pearson/Spearman correlations and conditional pass rates.
- Training-only cross-validation, optional sigmoid calibration and held-out evaluation.
- Research evidence page with a dataset mapping, experiment diagram, findings and downloadable aggregate results.
- 95% Wilson intervals for accuracy, recall and conditional pass rates; paired bootstrap intervals for Brier error and calibration comparisons.

## Two data workflows

**Research datasets (work immediately):** separate real [UCI Student Performance](https://doi.org/10.24432/C5TG7T) Portuguese-language and mathematics course cohorts. Portuguese is the default because it has more records and better held-out accuracy/probability error; Mathematics remains selectable because it identifies failing students more reliably in this fixed test split. Inputs are absences, study-time band, first-period marks and second-period marks. Pass is defined as final grade ≥10/20.

**Your four-factor CSV:** attendance (%), study hours/week, previous marks (%) and assignment score (%), with historical final marks. Upload anonymous records in the app to train and use this exact four-factor model. The pass mark defaults to 50/100 and is configurable before training.

**Data limitation:** UCI does not contain assignment scores, attendance percentages or exact study hours. These are not invented or substituted. The complete four-factor workflow requires your own compatible historical data. No synthetic research dataset is included.

## Run locally

From the repository root, using the existing virtual environment:

```powershell
.\student-performance-prediction\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

For a fresh installation, open `student-performance-prediction` and run `setup_windows.bat`, then `run_app.bat`.

See the [complete setup and research guide](student-performance-prediction/README.md) and [generated report](student-performance-prediction/reports/research_report.md).

The premium Probability Atlas interface is documented in [DESIGN.md](DESIGN.md). The complete [architecture audit, visual QA, and deployment report](docs/UI_UX_AUDIT.md) records what was preserved and improved.

Research foundation: [Cortez and Silva (2008), *Using Data Mining to Predict Secondary School Student Performance*](https://www3.dsi.uminho.pt/pcortez/student.pdf). The project extends that source with an explicitly probabilistic, calibrated Naive Bayes workflow and uncertainty reporting.

## Verified research results

Default model: 649 real Portuguese-language records, with 519 used for training and 130 held out for testing. Selected model: **calibrated categorical Naive Bayes**.

- Accuracy: 90.8%
- Fail recall: 75.0%; pass recall: 93.6%
- ROC-AUC: 0.951
- Brier score: 0.070 (raw Bayes: 0.076; training-prior baseline: 0.130)

The selectable Mathematics model remains at 87.3% accuracy with 96.2% fail recall. Reporting both prevents the higher overall accuracy of the more imbalanced Portuguese cohort from hiding weaker fail detection.

These are retrospective results for the UCI inputs, not accuracy claims for the four-factor CSV model. Absence timing is unspecified, prior-period grades are strong predictors, and local/future-cohort validation is needed.

## Streamlit Community Cloud

- Repository: `mirza9037/student-performance-prediction`
- Branch: `master`
- Main file: `streamlit_app.py`
- Python: **3.12**
- Secrets: none
- Set Sharing to **This app is public and searchable** for anonymous access.

The root requirements pin the trained model's runtime. The app loads the bundled model without retraining. GitHub source can remain private if Streamlit is authorized to access it. A GitHub push is not a live deployment.

Custom uploads are processed in session memory and are not written to disk. Do not include student identifiers. The previous dropout implementation remains available in Git history at commit `6db7858`.
