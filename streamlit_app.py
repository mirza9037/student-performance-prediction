"""Community Cloud entrypoint; reuse the tested application in the project folder."""
from pathlib import Path
import runpy
import sys

PROJECT = Path(__file__).resolve().parent / "student-performance-prediction"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

runpy.run_path(str(PROJECT / "app.py"), run_name="__main__")
