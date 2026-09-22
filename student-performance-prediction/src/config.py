"""Shared paths and explicitly allowed, stage-specific predictors."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "uci_student_dropout.csv"
MODEL_PATH = ROOT / "models" / "model_bundle.joblib"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
SEED = 42
FOLDS = 5
THRESHOLD = 0.50
DATA_URL = "https://archive.ics.uci.edu/static/public/697/data.csv"
ZIP_URL = "https://archive.ics.uci.edu/static/public/697/predict%2Bstudents%2Bdropout%2Band%2Bacademic%2Bsuccess.zip"
ENROLLMENT = [
    "Application order", "Previous qualification (grade)", "Admission grade",
    "Daytime/evening attendance", "Displaced", "Educational special needs",
    "Debtor", "Tuition fees up to date", "Scholarship holder", "Age at enrollment",
    "International", "Unemployment rate", "Inflation rate", "GDP",
]
SEMESTER = [f"Curricular units 1st semester ({name})" for name in
            ["credited", "enrolled", "evaluations", "approved", "grade", "without evaluations"]]
STAGES = {"enrollment": ENROLLMENT, "first_semester": ENROLLMENT + SEMESTER}
STAGE_LABELS = {"enrollment": "Enrollment", "first_semester": "First-Semester Early Warning"}
BINARY_FEATURES = ["Daytime/evening attendance", "Displaced", "Educational special needs",
                   "Debtor", "Tuition fees up to date", "Scholarship holder", "International"]
RESEARCH_QUESTIONS = [
    "Which academic and enrollment factors most strongly predict student risk?",
    "Which model gives the most reliable probabilities?",
    "Does first-semester information improve prediction over enrollment information?",
    "Are the model’s predictions equally reliable across relevant student groups?",
    "Can a model trained on international data be responsibly used in Pakistan without local validation?",
]
