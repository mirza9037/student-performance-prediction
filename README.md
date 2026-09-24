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

## Streamlit Community Cloud deployment

The repository includes a cloud entrypoint and runtime dependencies. In Streamlit Community Cloud, deploy with:

- Repository: `mirza9037/student-performance-prediction`
- Branch: `master`
- Main file path: `streamlit_app.py`
- Python version: **3.12** in Advanced settings
- Secrets: none required

The root `requirements.txt` pins the numerical and model libraries to the versions used for the committed trained models. The entrypoint reuses the existing application; deployment loads the models without retraining. The research/development requirements remain inside the project folder.

For anonymous internet access, set the deployed app's **Settings → Sharing → Who can view this app** to **This app is public and searchable**. The GitHub source repository can remain private. A successful GitHub push alone does not publish a running app.

After deployment, verify both prediction modes, all four pages, and public access without a Streamlit login. Public users should enter only anonymous information; the application does not persist submitted profiles to disk.

## Verified results

- Enrollment: calibrated Random Forest; test average precision 0.731, recall 52.1%, Brier score 0.153.
- First semester: calibrated HistGradientBoosting; test average precision 0.864, recall 69.0%, Brier score 0.104.
- Both models were selected using training-only cross-validation and evaluated on the same 885 held-out students. Recall uses threshold 0.50.
- All 21 tests passed, including the cloud entrypoint; Python syntax and live Streamlit startup were verified.

## Dataset and responsible use

The [UCI Predict Students' Dropout and Academic Success dataset](https://doi.org/10.24432/C5MC89) is provided under CC BY 4.0. Dropout is the positive class; Graduate and Enrolled form the negative class. Gender and second-semester variables are excluded from prediction.

This is a research prototype. Unresolved outcomes, prediction-time data availability, subgroup differences, and transfer from Portuguese data to Pakistani institutions require further investigation. Local validation and human review are required before institutional deployment.
