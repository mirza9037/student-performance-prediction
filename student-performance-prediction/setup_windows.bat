@echo off
setlocal
cd /d "%~dp0"
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"
if errorlevel 1 (
  echo Python 3.11 or newer is required. Install it and enable Add Python to PATH.
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" python -m venv .venv
if errorlevel 1 exit /b 1
call ".venv\Scripts\activate.bat"
if errorlevel 1 exit /b 1
python -m pip install --upgrade pip
if errorlevel 1 exit /b 1
python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
python train.py
if errorlevel 1 exit /b 1
echo Setup and training complete. Double-click run_app.bat to launch the dashboard.
echo Or run: python -m streamlit run app.py
endlocal
