@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
  echo Virtual environment missing. Run setup_windows.bat first.
  exit /b 1
)
call ".venv\Scripts\activate.bat"
if errorlevel 1 exit /b 1
if not exist "models\model_bundle.joblib" (
  echo Model missing. Run setup_windows.bat or python train.py first.
  exit /b 1
)
python -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
endlocal
