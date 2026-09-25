# AI-Based Student Performance Prediction Using Probability

Predict **pass and fail probabilities** with Naive Bayes, explore student factors, and see the calculation.

This revision replaces the earlier dropout system. It includes:
- Complementary pass/fail probabilities and a clear likely-pass/likely-fail classification.
- Interactive input changes and one-factor what-if charts.
- A worked Bayes calculation with priors, conditional likelihoods and posterior probabilities.
- Means, sample standard deviations, Pearson/Spearman correlations and conditional pass rates.
- Training-only cross-validation, optional sigmoid calibration and held-out evaluation.

## Two data workflows

**Research dataset (works immediately):** real [UCI Student Performance](https://doi.org/10.24432/C5TG7T) mathematics records. Inputs are absences, study-time band, first-period marks and second-period marks. Pass is defined as final grade ≥10/20.

**Your four-factor CSV:** attendance (%), study hours/week, previous marks (%) and assignment score (%), with historical final marks. Upload anonymous records in the app to train and use this exact four-factor model. The pass mark defaults to 50/100 and is configurable before training.

**Data limitation:** UCI does not contain assignment scores, attendance percentages or exact study hours. These are not invented or substituted. The complete four-factor workflow requires your own compatible historical data. No synthetic research dataset is included.

## Run locally

From the repository root, using the existing virtual environment:

```powershell
.\student-performance-prediction\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

For a fresh installation, open `student-performance-prediction` and run `setup_windows.bat`, then `run_app.bat`.

See the [complete setup and research guide](student-performance-prediction/README.md) and [generated report](student-performance-prediction/reports/research_report.md).

## Verified research results

395 real mathematics records, with 316 used for training and 79 held out for testing. Selected model: **calibrated categorical Naive Bayes**.

- Accuracy: 87.3%
- Fail recall: 96.2%; pass recall: 83.0%
- ROC-AUC: 0.905
- Brier score: 0.112 (raw Bayes: 0.121; training-prior baseline: 0.221)

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
