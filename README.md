# Parkinson's Detection from Vocal Biomarkers - Streamlit Demo

Interactive demo for the paper *"Boosting Ensemble Learning for Early Parkinson's Disease
Detection using Vocal Biomarkers"* (ICICT-2026).

> Research demo only. Not a medical device and not a diagnosis.

## Features
- Predict on any row of the UCI dataset (shows whether the row was held out for testing)
- Enter all 22 acoustic features manually
- Record or upload a sustained vowel; Praat extracts 16 features and a lite model predicts
- Switch between LightGBM, XGBoost and CatBoost
- Paper results table and charts, plus this app's hold-out metrics and feature importance

## Setup
```bash
pip install -r requirements.txt

# 1. Download parkinsons.data from
#    https://archive.ics.uci.edu/dataset/174/parkinsons
#    and put it in this folder
# 2. Train (creates the artifacts/ folder)
python train_model.py --data parkinsons.data

# 3. Run the app
streamlit run app.py
```

## Project structure
| File | Purpose |
|---|---|
| `train_model.py` | Min-Max scaling, 80/20 stratified split (seed 42), trains the three models for the 22-feature and 16-feature versions |
| `audio_features.py` | Praat (parselmouth) feature extraction from audio |
| `app.py` | Streamlit interface |
| `artifacts/` | Saved models, scaler, metrics and sample rows (created by training) |

## Deploy free
Push the folder (including `artifacts/`) to GitHub, then create an app on
[Streamlit Community Cloud](https://streamlit.io/cloud) pointing at `app.py`.
If you change scikit-learn versions, re-run `train_model.py` so the saved files match.

## Limitations
195 recordings from 31 people, class imbalance (147 vs 48), possible subject overlap between
train and test, and a domain gap between UCI's MDVP-derived features and Praat features.
