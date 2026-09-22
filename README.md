# AI-Based Student Performance Prediction Using Probability

A research-based Python project that estimates calibrated student dropout probabilities using the official UCI dataset. It includes enrollment and first-semester prediction models, a four-page Streamlit dashboard, a research report, fairness diagnostics, and reproducible tests.

The complete project is in **[student-performance-prediction/](student-performance-prediction/)**. See its [installation and usage guide](student-performance-prediction/README.md) and [research report](student-performance-prediction/reports/research_report.md).

## Run on Windows

With Python 3.11 or newer installed:

```powershell
cd student-performance-prediction
.\setup_windows.bat
.\run_app.bat
```

The virtual environment is intentionally excluded from this repository. Setup installs dependencies and regenerates the research outputs. Pretrained models and the original experiment's results are included for inspection.

## Verified results

- Enrollment: calibrated Random Forest; test average precision 0.731, recall 52.1%, Brier score 0.153.
- First semester: calibrated HistGradientBoosting; test average precision 0.864, recall 69.0%, Brier score 0.104.
- Both models were selected using training-only cross-validation and evaluated on the same 885 held-out students. Recall uses threshold 0.50.
- All 20 tests passed; Python syntax and live Streamlit startup were verified.

## Dataset and responsible use

The [UCI Predict Students' Dropout and Academic Success dataset](https://doi.org/10.24432/C5MC89) is provided under CC BY 4.0. Dropout is the positive class; Graduate and Enrolled form the negative class. Gender and second-semester variables are excluded from prediction.

This is a research prototype. Unresolved outcomes, prediction-time data availability, subgroup differences, and transfer from Portuguese data to Pakistani institutions require further investigation. Local validation and human review are required before institutional deployment.
