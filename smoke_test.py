import joblib, pandas as pd
from src import explain, assistant
from src.config import FEATURES
a = joblib.load("models/artifacts.joblib")
row = a["X_test"].iloc[[0]]
res = {"model": a["best_name"], "probability": float(a["pipeline"].predict_proba(row)[0,1]),
       "local": explain.local_explanation(a["pipeline"], row, a["train_medians"]), "flags": explain.flag_values(row)}
print(explain.global_importance(a["pipeline"], a["X_test"], a["y_test"], n_repeats=5).head(4)[["label","importance"]])
print(res["local"].head(3))
print(assistant.answer("why?", res)[:300]); print(assistant.answer("what is ALT", res)[:80])
print(len(assistant.build_report(row.iloc[0].to_dict(), res)))
