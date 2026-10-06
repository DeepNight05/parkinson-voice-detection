"""Train the models used by the Streamlit app.

Mirrors the setup in the paper: Min-Max scaling, stratified 80/20 split,
seed 42, and the same hyperparameters for XGBoost / LightGBM / CatBoost.

Two model bundles are produced:
  * full  - all 22 UCI features (used for sample rows and manual entry)
  * lite  - the 16 features that Praat can compute from a raw recording
            (used for the "upload your voice" tab)

Usage:
    python train_model.py --data parkinsons.data --out artifacts
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

SEED = 42
LABEL = "status"

# Features that can be measured directly from an audio file with Praat.
PRAAT_FEATURES = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR",
]


def make_models():
    return {
        "LightGBM": LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,
                                   num_leaves=31, random_state=SEED, verbose=-1),
        "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,
                                 eval_metric="logloss", random_state=SEED),
        "CatBoost": CatBoostClassifier(iterations=100, learning_rate=0.1, depth=3,
                                       verbose=0, random_seed=SEED),
    }


def fit_bundle(df, features, train_idx, test_idx):
    X = df[features].to_numpy(dtype=float)
    y = df[LABEL].to_numpy(dtype=int)

    scaler = MinMaxScaler().fit(X[train_idx])
    Xtr, Xte = scaler.transform(X[train_idx]), scaler.transform(X[test_idx])
    ytr, yte = y[train_idx], y[test_idx]

    models, metrics = make_models(), {}
    for name, model in models.items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        proba = model.predict_proba(Xte)[:, 1]
        metrics[name] = {
            "accuracy": accuracy_score(yte, pred),
            "precision": precision_score(yte, pred, zero_division=0),
            "recall": recall_score(yte, pred, zero_division=0),
            "f1": f1_score(yte, pred, zero_division=0),
            "roc_auc": roc_auc_score(yte, proba),
        }

    ranges = {f: {"min": float(df[f].min()), "max": float(df[f].max()),
                  "median": float(df[f].median())} for f in features}
    bundle = {"features": features, "scaler": scaler, "models": models, "ranges": ranges}
    return bundle, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="parkinsons.data")
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    full_features = [c for c in df.columns if c not in ("name", LABEL)]
    missing = [f for f in PRAAT_FEATURES if f not in df.columns]
    if missing:
        raise SystemExit(f"Dataset is missing expected columns: {missing}")

    # One shared split so both bundles are evaluated on the same test rows.
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.2, stratify=df[LABEL], random_state=SEED)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    full, full_metrics = fit_bundle(df, full_features, train_idx, test_idx)
    lite, lite_metrics = fit_bundle(df, PRAAT_FEATURES, train_idx, test_idx)

    joblib.dump(full, out / "bundle_full.joblib")
    joblib.dump(lite, out / "bundle_lite.joblib")
    (out / "metrics.json").write_text(json.dumps(
        {"full_22_features": full_metrics, "lite_16_features": lite_metrics,
         "n_train": int(len(train_idx)), "n_test": int(len(test_idx))}, indent=2))

    samples = df.copy()
    samples["split"] = "train"
    samples.loc[test_idx, "split"] = "test"
    samples.to_csv(out / "samples.csv", index=False)

    print(f"Saved artifacts to {out}/")
    for tag, m in (("full (22 features)", full_metrics), ("lite (16 features)", lite_metrics)):
        print(f"\n{tag} - hold-out test set ({len(test_idx)} rows)")
        for name, v in m.items():
            print(f"  {name:9s} acc={v['accuracy']:.3f} prec={v['precision']:.3f} "
                  f"rec={v['recall']:.3f} f1={v['f1']:.3f} auc={v['roc_auc']:.3f}")


if __name__ == "__main__":
    main()
