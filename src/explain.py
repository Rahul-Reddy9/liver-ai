"""Explainable AI for the PCA-based pipeline.

SHAP on PCA components would only tell us about "PC1, PC2 ..." which nobody
can interpret. Instead we explain in terms of the ORIGINAL clinical features,
using two model-agnostic methods applied to the *whole* pipeline
(impute -> scale -> PCA -> classifier):

  * Global:  permutation importance on the held-out test set
  * Local :  per-patient occlusion -- replace one feature at a time with the
             training median and measure how much P(disease) changes.
"""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .config import FEATURES, FEATURE_INFO


def pretty(f):
    return FEATURE_INFO[f][0]


def global_importance(pipeline, X_test, y_test, n_repeats=20, seed=42) -> pd.DataFrame:
    r = permutation_importance(pipeline, X_test, y_test, scoring="roc_auc",
                               n_repeats=n_repeats, random_state=seed)
    out = pd.DataFrame({"feature": FEATURES, "importance": r.importances_mean,
                        "std": r.importances_std})
    out["label"] = out["feature"].map(pretty)
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def local_explanation(pipeline, patient: pd.DataFrame, medians: dict) -> pd.DataFrame:
    """Positive contribution = this feature value pushes toward 'liver disease'."""
    patient = patient[FEATURES].astype(float)
    base = pipeline.predict_proba(patient)[0, 1]
    rows = []
    for f in FEATURES:
        tmp = patient.copy()
        tmp[f] = medians[f]
        p = pipeline.predict_proba(tmp)[0, 1]
        rows.append({"feature": f, "label": pretty(f), "value": float(patient[f].iloc[0]),
                     "contribution": base - p})
    out = pd.DataFrame(rows)
    out["abs"] = out["contribution"].abs()
    return out.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)


def flag_values(patient: pd.DataFrame) -> list[dict]:
    """Compare each lab value with its typical reference range."""
    flags = []
    for f in FEATURES:
        name, unit, rng = FEATURE_INFO[f]
        if rng is None:
            continue
        v = float(patient[f].iloc[0])
        status = "low" if v < rng[0] else "high" if v > rng[1] else "normal"
        flags.append({"feature": f, "name": name, "value": v, "unit": unit,
                      "range": rng, "status": status})
    return flags
