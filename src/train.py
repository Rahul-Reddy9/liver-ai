"""Train + compare the six classifiers on PCA-reduced data.

Run:  python -m src.train
Pipeline per model:  median-imputer -> StandardScaler -> PCA -> classifier
(All fitted on the training split only, so there is no data leakage.)
"""
import json, time
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import ARTIFACT_PATH, FEATURES, METRICS_PATH, MODEL_DIR
from .data_utils import load_clean, split_xy

SEED = 42
PCA_VARIANCE = 0.95   # keep enough components to explain 95% of the variance

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED),
    "SVM": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
}


def build_pipeline(clf):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=PCA_VARIANCE, random_state=SEED)),
        ("clf", clf),
    ])


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    df = load_clean()
    X, y = split_xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    print(f"Rows: {len(df)} | train {len(X_tr)} | test {len(X_te)} | "
          f"disease rate {y.mean():.1%}")

    results, fitted = {}, {}
    for name, clf in MODELS.items():
        pipe = build_pipeline(clf)
        t0 = time.perf_counter(); pipe.fit(X_tr, y_tr); train_t = time.perf_counter() - t0
        t0 = time.perf_counter(); pred = pipe.predict(X_te); pred_t = time.perf_counter() - t0
        proba = pipe.predict_proba(X_te)[:, 1]
        results[name] = {
            "accuracy": accuracy_score(y_te, pred),
            "precision": precision_score(y_te, pred, zero_division=0),
            "recall": recall_score(y_te, pred),
            "f1": f1_score(y_te, pred),
            "roc_auc": roc_auc_score(y_te, proba),
            "train_time_s": train_t,
            "predict_time_s": pred_t,
            "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
        }
        fitted[name] = pipe
        r = results[name]
        print(f"{name:20s} acc={r['accuracy']:.3f} prec={r['precision']:.3f} "
              f"rec={r['recall']:.3f} f1={r['f1']:.3f} auc={r['roc_auc']:.3f}")

    # Best model = highest F1 (balances precision & recall for the disease class)
    best_name = max(results, key=lambda k: results[k]["f1"])
    best = fitted[best_name]
    pca = best.named_steps["pca"]
    print(f"\nBest model: {best_name}  |  PCA components kept: {pca.n_components_} of {len(FEATURES)}")

    pca_info = {
        "n_components": int(pca.n_components_),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
        # loadings: how each original feature contributes to each component
        "loadings": pca.components_.tolist(),
    }
    joblib.dump({
        "best_name": best_name,
        "pipeline": best,
        "all_pipelines": fitted,
        "features": FEATURES,
        "X_train": X_tr, "y_train": y_tr,
        "X_test": X_te, "y_test": y_te,
        "train_medians": X_tr.median().to_dict(),
    }, ARTIFACT_PATH)
    METRICS_PATH.write_text(json.dumps(
        {"best_model": best_name, "results": results, "pca": pca_info,
         "n_train": len(X_tr), "n_test": len(X_te)}, indent=2))
    print("Saved:", ARTIFACT_PATH, "and", METRICS_PATH)


if __name__ == "__main__":
    main()
