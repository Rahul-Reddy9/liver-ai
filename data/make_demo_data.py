"""Generates a SYNTHETIC stand-in with the same schema as the ILPD dataset so
the project runs end-to-end. Replace data/indian_liver_patient.csv with the
real dataset (Kaggle / UCI 'ILPD') for genuine results."""
import numpy as np, pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
n_pos, n_neg = 416, 167
def lognorm(mu, sig, n): return np.exp(rng.normal(mu, sig, n))

def block(n, sick):
    k = 1 if sick else 0
    d = pd.DataFrame({
        "Age": np.clip(rng.normal(46 + 2*k, 16, n), 4, 90).round(),
        "Gender": rng.choice(["Male", "Female"], n, p=[.76, .24] if sick else [.62, .38]),
        "Total_Bilirubin": lognorm(-0.1 + 1.0*k, 0.8 + .3*k, n),
    })
    d["Direct_Bilirubin"] = d["Total_Bilirubin"] * rng.uniform(.25, .55, n)
    d["Alkaline_Phosphotase"] = lognorm(5.2 + .35*k, .45, n)
    d["Alamine_Aminotransferase"] = lognorm(3.2 + .9*k, .7 + .2*k, n)
    d["Aspartate_Aminotransferase"] = lognorm(3.4 + .9*k, .7 + .2*k, n)
    d["Total_Protiens"] = rng.normal(6.6 - .25*k, .9, n)
    d["Albumin"] = rng.normal(3.4 - .3*k, .7, n)
    d["Albumin_and_Globulin_Ratio"] = d["Albumin"] / np.clip(d["Total_Protiens"] - d["Albumin"], .6, None)
    d["Dataset"] = 1 if sick else 2
    return d

df = pd.concat([block(n_pos, True), block(n_neg, False)]).sample(frac=1, random_state=1).round(2)
df.loc[df.sample(4, random_state=3).index, "Albumin_and_Globulin_Ratio"] = np.nan  # like real ILPD
out = Path(__file__).parent / "indian_liver_patient.csv"
df.to_csv(out, index=False)
print("wrote", out, df.shape)
