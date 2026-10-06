import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ART = Path(os.environ.get("PD_ARTIFACTS", Path(__file__).parent / "artifacts"))

st.set_page_config(page_title="Parkinson's Voice Screening Demo",
                   page_icon="🎙️", layout="wide")

# Results reported in the paper (Table I, 10-fold stratified CV).
PAPER = pd.DataFrame({
    "Model": ["LightGBM", "XGBoost", "CatBoost", "AdaBoost", "LinearBoost"],
    "Accuracy": [0.93263, 0.93237, 0.92763, 0.91763, 0.89158],
    "F1-score": [0.95558, 0.95631, 0.95214, 0.94702, 0.93012],
    "Precision": [0.94655, 0.94167, 0.94696, 0.92943, 0.90651],
    "Recall": [0.96571, 0.97238, 0.95905, 0.96619, 0.95905],
    "ROC-AUC": [0.97233, 0.97090, 0.97557, 0.97176, 0.92924],
})


@st.cache_resource
def load_artifacts():
    needed = ["bundle_full.joblib", "bundle_lite.joblib", "metrics.json", "samples.csv"]
    if any(not (ART / n).exists() for n in needed):
        return None
    return (joblib.load(ART / "bundle_full.joblib"),
            joblib.load(ART / "bundle_lite.joblib"),
            json.loads((ART / "metrics.json").read_text()),
            pd.read_csv(ART / "samples.csv"))


def predict(bundle, model_name, values):
    x = np.array([[values[f] for f in bundle["features"]]], dtype=float)
    xs = bundle["scaler"].transform(x)
    return float(bundle["models"][model_name].predict_proba(xs)[0, 1])


def show_result(p):
    label = "Parkinson's-like voice pattern" if p >= 0.5 else "Healthy-like voice pattern"
    c1, c2 = st.columns(2)
    c1.metric("Model output", label)
    c2.metric("Probability of Parkinson's pattern", f"{p:.1%}")
    st.progress(min(max(p, 0.0), 1.0))


artifacts = load_artifacts()
if artifacts is None:
    st.error("Model files not found. Put `parkinsons.data` in this folder and run "
             "`python train_model.py` first (see README).")
    st.stop()
full, lite, metrics, samples = artifacts

st.title("🎙️ Early Parkinson's Detection from Vocal Biomarkers")
st.caption("Demo of the boosting-ensemble comparison from our ICICT-2026 paper. "
           "This is a research demo, **not a medical device or diagnosis**.")

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Model", ["LightGBM", "XGBoost", "CatBoost"],
                              help="LightGBM was the best overall model in the paper.")
    st.markdown("**Pipeline**\n\nMin-Max scaling → boosting classifier → probability")
    st.warning("Trained on only 195 recordings from 31 people. "
               "Do not use for any health decision.")

tab_sample, tab_manual, tab_audio, tab_results, tab_about = st.tabs(
    ["Try a dataset sample", "Enter values", "Upload your voice",
     "Model comparison", "About & limitations"])

# ---------------------------------------------------------------- sample rows
with tab_sample:
    st.subheader("Pick a recording from the UCI dataset")
    row_id = st.number_input("Row number", 0, len(samples) - 1, 0, step=1)
    row = samples.iloc[int(row_id)]
    truth = "Parkinson's" if int(row["status"]) == 1 else "Healthy"
    st.write(f"**Recording:** `{row['name']}`  |  **True label:** {truth}  |  "
             f"**Used for** {'testing (unseen by the model)' if row['split'] == 'test' else 'training'}")
    values = {f: float(row[f]) for f in full["features"]}
    with st.expander("Show the 22 voice features"):
        st.dataframe(pd.DataFrame(values, index=["value"]).T)
    show_result(predict(full, model_name, values))
    if row["split"] == "train":
        st.caption("This row was in the training split, so a correct prediction here "
                   "proves little. Pick a 'testing' row for a fairer check.")

# --------------------------------------------------------------- manual entry
with tab_manual:
    st.subheader("Enter the 22 acoustic features")
    st.caption("Defaults are the dataset medians. Change any value and predict.")
    vals = {}
    cols = st.columns(4)
    for i, f in enumerate(full["features"]):
        r = full["ranges"][f]
        vals[f] = cols[i % 4].number_input(f, value=r["median"], format="%.5f",
                                            help=f"Dataset range: {r['min']:.5f} to {r['max']:.5f}")
    if st.button("Predict", type="primary", key="manual_predict"):
        show_result(predict(full, model_name, vals))

# ------------------------------------------------------------------ audio tab
with tab_audio:
    st.subheader("Record or upload a sustained vowel")
    st.write("Say a steady **'aaah'** for 3-5 seconds in a quiet room, or upload a WAV file. "
             "Praat measures 16 of the 22 features from the audio, and a model trained on "
             "those 16 features makes the prediction.")
    st.info("Demo only: the dataset was recorded with different software and equipment, "
            "so results from your own microphone are not clinically meaningful.")

    audio = None
    if hasattr(st, "audio_input"):
        audio = st.audio_input("Record your voice")
    uploaded = st.file_uploader("...or upload a WAV file", type=["wav"])
    audio = audio or uploaded

    if audio is not None:
        try:
            from audio_features import extract_features
            feats = extract_features(audio.getvalue())
        except Exception as e:  # show friendly message instead of a stack trace
            st.error(str(e))
        else:
            st.dataframe(pd.DataFrame(feats, index=["measured"]).T)
            show_result(predict(lite, model_name, feats))
            m = metrics["lite_16_features"][model_name]
            st.caption(f"This 16-feature model scored {m['accuracy']:.1%} accuracy on the "
                       f"hold-out set of {metrics['n_test']} dataset rows.")

# --------------------------------------------------------------- results tab
with tab_results:
    st.subheader("Results reported in the paper (10-fold stratified CV)")
    st.dataframe(PAPER.set_index("Model").style.format("{:.3%}")
                 .highlight_max(axis=0, color="#cfe8cf"), width="stretch")
    st.bar_chart(PAPER.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]])
    st.markdown("- **LightGBM**: best accuracy\n- **XGBoost**: best recall and F1 (fewest missed patients)\n"
                "- **CatBoost**: best precision and ROC-AUC (fewest false alarms)")

    st.subheader("This app's models on a single 80/20 hold-out split")
    rows = []
    for tag, key in (("22 features", "full_22_features"), ("16 features (audio)", "lite_16_features")):
        for name, v in metrics[key].items():
            rows.append({"Feature set": tag, "Model": name, **{k: v[k] for k in
                         ("accuracy", "precision", "recall", "f1", "roc_auc")}})
    st.dataframe(pd.DataFrame(rows).style.format(
        {c: "{:.3f}" for c in ("accuracy", "precision", "recall", "f1", "roc_auc")}),
        width="stretch", hide_index=True)
    st.caption(f"Test set is only {metrics['n_test']} rows, so these numbers are noisy.")

    st.subheader(f"Feature importance - {model_name}")
    model = full["models"][model_name]
    imp = pd.Series(model.feature_importances_, index=full["features"]).sort_values(ascending=False)
    st.bar_chart(imp.head(10))

# ------------------------------------------------------------------ about tab
with tab_about:
    st.markdown("""
### What this is
A small demo around our paper comparing five boosting algorithms (XGBoost, CatBoost,
LightGBM, AdaBoost, LinearBoost) for detecting Parkinson's disease from voice features
in the Oxford/UCI Parkinson's dataset.

### Honest limitations
- **Small dataset:** 195 recordings from 31 people (23 with Parkinson's, 8 healthy).
- **Possible subject overlap:** each person has about 6 recordings, so a random split can
  put the same person in both train and test and inflate scores.
- **Class imbalance:** 147 Parkinson's vs 48 healthy recordings.
- **Domain shift:** your microphone and Praat features differ from the original recording setup.
- **Not a diagnosis:** only a clinician can diagnose Parkinson's disease.

### Future work
Subject-wise cross-validation, larger datasets, SHAP explanations, deep learning models.
""")
