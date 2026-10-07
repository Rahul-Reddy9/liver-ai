"""Loading + cleaning the liver dataset (ILPD: Indian Liver Patient Dataset)."""
import pandas as pd
from .config import DATA_PATH, FEATURES, TARGET

_RENAME = {
    "Dataset": TARGET, "Selector": TARGET, "is_patient": TARGET,
    "Total_Proteins": "Total_Protiens",
    "Alkaline_Phosphatase": "Alkaline_Phosphotase",
    "Alanine_Aminotransferase": "Alamine_Aminotransferase",
}


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    # UCI raw file has no header row -> re-read with explicit names
    if "Age" not in df.columns:
        df = pd.read_csv(path, header=None, names=FEATURES + [TARGET])
    df = df.rename(columns=_RENAME)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    return df[FEATURES + [TARGET]]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categoricals, fix target coding, remove duplicates.
    Missing numeric values are imputed later *inside* the sklearn pipeline
    (fit on train only) to avoid data leakage."""
    df = df.copy()
    df["Gender"] = (df["Gender"].astype(str).str.strip().str.lower()
                    .map({"male": 1, "m": 1, "female": 0, "f": 0}))
    # ILPD codes: 1 = liver patient, 2 = non-patient  ->  1 / 0
    df[TARGET] = (df[TARGET] == 1).astype(int)
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.drop_duplicates().reset_index(drop=True)


def load_clean(path=DATA_PATH) -> pd.DataFrame:
    return clean(load_raw(path))


def split_xy(df: pd.DataFrame):
    return df[FEATURES], df[TARGET]
