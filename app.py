"""Streamlit dashboard:  streamlit run app.py"""
import json
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src import assistant, explain
from src.config import ARTIFACT_PATH, FEATURES, FEATURE_INFO, METRICS_PATH

st.set_page_config(page_title="AI Liver Disease Prediction", page_icon="🩺", layout="wide")


@st.cache_resource
def load_artifacts():
    if not ARTIFACT_PATH.exists():
        return None, None
    return joblib.load(ARTIFACT_PATH), json.loads(METRICS_PATH.read_text())


@st.cache_data
def cached_global_importance(_art):
    return explain.global_importance(_art["pipeline"], _art["X_test"], _art["y_test"])


art, metrics = load_artifacts()
if art is None:
    st.error("No trained model found. Run `python -m src.train` first.")
    st.stop()

st.title("🩺 AI-Based Liver Disease Prediction")
st.caption("PCA + Machine Learning + Explainable AI + AI Liver Health Assistant")
st.sidebar.success(f"Best model: **{art['best_name']}**")
page = st.sidebar.radio("Page", ["1 · Input & Prediction", "2 · Explanation (XAI)",
                                 "3 · AI Assistant", "4 · Model Comparison", "5 · Report"])
st.sidebar.warning(assistant.DISCLAIMER)

med = art["train_medians"]

# ---------------- Input form (kept in session state so every page can use it) ----
def get_inputs():
    d = st.session_state.get("inputs")
    return d if d else {f: (1 if f == "Gender" else med[f]) for f in FEATURES}


if page.startswith("1"):
    st.header("Enter patient parameters")
    cur = get_inputs()
    c1, c2, c3 = st.columns(3)
    vals = {}
    with c1:
        vals["Age"] = st.number_input("Age (years)", 1, 100, int(cur["Age"]))
        g = st.selectbox("Gender", ["Male", "Female"], index=0 if cur["Gender"] == 1 else 1)
        vals["Gender"] = 1 if g == "Male" else 0
        vals["Total_Bilirubin"] = st.number_input("Total Bilirubin (mg/dL)", 0.0, 80.0, float(cur["Total_Bilirubin"]), 0.1)
        vals["Direct_Bilirubin"] = st.number_input("Direct Bilirubin (mg/dL)", 0.0, 40.0, float(cur["Direct_Bilirubin"]), 0.1)
    with c2:
        vals["Alkaline_Phosphotase"] = st.number_input("Alkaline Phosphatase (IU/L)", 0.0, 2500.0, float(cur["Alkaline_Phosphotase"]), 1.0)
        vals["Alamine_Aminotransferase"] = st.number_input("ALT / SGPT (IU/L)", 0.0, 3000.0, float(cur["Alamine_Aminotransferase"]), 1.0)
        vals["Aspartate_Aminotransferase"] = st.number_input("AST / SGOT (IU/L)", 0.0, 5000.0, float(cur["Aspartate_Aminotransferase"]), 1.0)
    with c3:
        vals["Total_Protiens"] = st.number_input("Total Proteins (g/dL)", 0.0, 12.0, float(cur["Total_Protiens"]), 0.1)
        vals["Albumin"] = st.number_input("Albumin (g/dL)", 0.0, 7.0, float(cur["Albumin"]), 0.1)
        vals["Albumin_and_Globulin_Ratio"] = st.number_input("Albumin/Globulin Ratio", 0.0, 4.0, float(cur["Albumin_and_Globulin_Ratio"]), 0.05)
    if st.button("Predict", type="primary"):
        st.session_state["inputs"] = vals
        patient = pd.DataFrame([vals])[FEATURES]
        p = float(art["pipeline"].predict_proba(patient)[0, 1])
        st.session_state["result"] = {
            "model": art["best_name"], "probability": p,
            "local": explain.local_explanation(art["pipeline"], patient, med),
            "flags": explain.flag_values(patient),
        }
    res = st.session_state.get("result")
    if res:
        p = res["probability"]
        (st.error if p >= 0.5 else st.success)(
            f"**Prediction: {'Liver Disease (higher likelihood)' if p >= 0.5 else 'No Liver Disease (lower likelihood)'}**"
            f" — estimated probability {p:.1%}")
        st.progress(min(max(p, 0.0), 1.0))
        st.caption("Go to the next pages for the explanation, assistant and report.")

else:
    res = st.session_state.get("result")

    if page.startswith("2"):
        st.header("Why this prediction?")
        gi = cached_global_importance(art)
        if res:
            loc = res["local"].copy()
            loc["direction"] = loc["contribution"].apply(lambda c: "Toward disease" if c > 0 else "Away from disease")
            fig = px.bar(loc.sort_values("contribution"), x="contribution", y="label", color="direction",
                         orientation="h", color_discrete_map={"Toward disease": "#d62728", "Away from disease": "#2ca02c"},
                         title="This patient — change in P(disease) when each feature is set to the typical value")
            fig.update_layout(yaxis_title="", xaxis_title="Contribution to probability")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Make a prediction on page 1 to see the patient-level explanation.")
        fig2 = px.bar(gi.sort_values("importance"), x="importance", y="label", error_x="std", orientation="h",
                      title="Overall model — permutation importance (drop in ROC-AUC on test data)")
        fig2.update_layout(yaxis_title="", xaxis_title="Importance")
        st.plotly_chart(fig2, width="stretch")
        st.caption("Explanations are computed on the original lab features through the full "
                   "scale → PCA → classifier pipeline, so they stay human-readable.")

    elif page.startswith("3"):
        st.header("AI Liver Health Assistant")
        if not res:
            st.info("Make a prediction on page 1 first.")
        else:
            if "chat" not in st.session_state:
                st.session_state["chat"] = [("assistant", assistant.answer("Explain my result in simple terms.", res))]
            q = st.chat_input("Ask e.g. 'Why did the model predict this?' or 'What is ALT?'")
            if q:
                st.session_state["chat"].append(("user", q))
                st.session_state["chat"].append(("assistant", assistant.answer(q, res)))
            for role, msg in st.session_state["chat"]:
                st.chat_message(role).markdown(msg)

    elif page.startswith("4"):
        st.header("Model comparison (PCA-reduced features)")
        rows = [{"Model": m, "Accuracy": r["accuracy"], "Precision": r["precision"], "Recall": r["recall"],
                 "F1": r["f1"], "ROC-AUC": r["roc_auc"], "Train time (s)": r["train_time_s"],
                 "Predict time (s)": r["predict_time_s"]} for m, r in metrics["results"].items()]
        df = pd.DataFrame(rows).set_index("Model")
        st.dataframe(df.style.format("{:.3f}").highlight_max(subset=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"], color="#c8e6c9"))
        long = df[["Accuracy", "Precision", "Recall", "F1"]].reset_index().melt("Model", var_name="Metric")
        st.plotly_chart(px.bar(long, x="Model", y="value", color="Metric", barmode="group"), width="stretch")

        st.subheader("Confusion matrices")
        cols = st.columns(3)
        for i, (m, r) in enumerate(metrics["results"].items()):
            cm = pd.DataFrame(r["confusion_matrix"], index=["Actual: No", "Actual: Yes"], columns=["Pred: No", "Pred: Yes"])
            with cols[i % 3]:
                st.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues", title=m), width="stretch")

        st.subheader("PCA")
        pca = metrics["pca"]
        ev = pd.DataFrame({"Component": [f"PC{i+1}" for i in range(pca["n_components"])],
                           "Explained variance": pca["explained_variance_ratio"],
                           "Cumulative": pca["cumulative_variance"]})
        st.write(f"{pca['n_components']} components kept out of {len(FEATURES)} features (≥95% variance).")
        st.plotly_chart(px.line(ev, x="Component", y=["Explained variance", "Cumulative"], markers=True), width="stretch")
        load = pd.DataFrame(pca["loadings"], columns=[FEATURE_INFO[f][0] for f in FEATURES],
                            index=ev["Component"])
        st.plotly_chart(px.imshow(load, color_continuous_scale="RdBu", zmin=-1, zmax=1, aspect="auto",
                                  title="PCA loadings (feature contribution to each component)"), width="stretch")

    elif page.startswith("5"):
        st.header("AI-generated report")
        if not res:
            st.info("Make a prediction on page 1 first.")
        else:
            report = assistant.build_report(st.session_state["inputs"], res)
            st.markdown(report)
            st.download_button("Download report (.md)", report, "liver_report.md", "text/markdown")
