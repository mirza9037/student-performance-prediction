"""Dataset contracts and paths for the pass/fail probability project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "student-mat.csv"
MODEL_PATH = ROOT / "models" / "model_bundle.joblib"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
SEED = 42
DATA_URL = "https://archive.ics.uci.edu/static/public/320/student%2Bperformance.zip"
TITLE = "AI-Based Student Performance Prediction Using Probability"

# Fixed intervals are declared in advance, never learned from test data.
SCHEMAS = {
    "uci": {
        "name": "UCI mathematics research dataset",
        "features": ["absences", "studytime", "G1", "G2"],
        "labels": ["School absences", "Weekly study-time band", "First-period marks (0–20)", "Second-period marks (0–20)"],
        "bounds": [(0, 93), (1, 4), (0, 20), (0, 20)],
        "bins": [[3, 10, 20], [1.5, 2.5, 3.5], [5, 10, 15], [5, 10, 15]],
        "defaults": [4, 2, 12, 12],
        "target": "G3", "pass_mark": 10.0, "target_max": 20.0,
        "limitation": "UCI records absences and study-time bands. It has no attendance percentage, exact study hours, or assignment scores. Previous marks are G1 and G2; neither is an assignment score.",
    },
    "custom": {
        "name": "Your four-factor dataset",
        "features": ["attendance", "study_hours", "previous_marks", "assignment_score"],
        "labels": ["Attendance (%)", "Study hours per week", "Previous marks (%)", "Assignment performance (%)"],
        "bounds": [(0, 100), (0, 168), (0, 100), (0, 100)],
        "bins": [[60, 75, 90], [5, 10, 20], [40, 60, 80], [40, 60, 80]],
        "defaults": [80, 10, 65, 70],
        "target": "final_marks", "pass_mark": 50.0, "target_max": 100.0,
        "limitation": "Use one anonymous row per student. All four inputs must be recorded before the final assessment. Model quality depends on the source and representativeness of your records.",
    },
}
STUDY_BANDS = {1: "Less than 2 hours/week", 2: "2–5 hours/week", 3: "5–10 hours/week", 4: "More than 10 hours/week"}
