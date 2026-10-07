"""Shared constants for the project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "indian_liver_patient.csv"
MODEL_DIR = ROOT / "models"
ARTIFACT_PATH = MODEL_DIR / "artifacts.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

FEATURES = [
    "Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin",
    "Alkaline_Phosphotase", "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase", "Total_Protiens", "Albumin",
    "Albumin_and_Globulin_Ratio",
]
TARGET = "Liver_Disease"  # 1 = liver disease, 0 = no liver disease

# Friendly names + units + typical adult reference ranges (approximate, for
# explanation only -- labs differ; this is NOT medical advice).
FEATURE_INFO = {
    "Age": ("Age", "years", None),
    "Gender": ("Gender", "", None),
    "Total_Bilirubin": ("Total Bilirubin", "mg/dL", (0.1, 1.2)),
    "Direct_Bilirubin": ("Direct Bilirubin", "mg/dL", (0.0, 0.3)),
    "Alkaline_Phosphotase": ("Alkaline Phosphatase (ALP)", "IU/L", (44, 147)),
    "Alamine_Aminotransferase": ("ALT (SGPT)", "IU/L", (7, 56)),
    "Aspartate_Aminotransferase": ("AST (SGOT)", "IU/L", (10, 40)),
    "Total_Protiens": ("Total Proteins", "g/dL", (6.0, 8.3)),
    "Albumin": ("Albumin", "g/dL", (3.5, 5.0)),
    "Albumin_and_Globulin_Ratio": ("Albumin/Globulin Ratio", "", (1.1, 2.5)),
}
