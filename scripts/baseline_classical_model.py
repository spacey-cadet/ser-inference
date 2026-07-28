"""
Track 1.1 — classical acoustic-feature baseline.

Extracts pitch, energy, MFCCs, zero-crossing rate with librosa on your
existing RAVDESS/IEMOCAP/CREMA-D data and fits a Logistic Regression /
Random Forest on top. Gives you a real "WavLM improves X points over a
classical baseline" number instead of an assertion. Same-afternoon, CPU-only.

Usage:
    python scripts/baseline_classical_model.py \
        --data-csv data/labels.csv \
        --audio-col path --label-col emotion \
        --out data/baseline_report.json

Expects a CSV with at least: path-to-wav, emotion-label columns.
"""
import argparse
import json

import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def extract_features(path: str) -> np.ndarray:
    y, sr = librosa.load(path, sr=16000, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    rms = librosa.feature.rms(y=y).mean()
    pitch = librosa.yin(y, fmin=50, fmax=400, sr=sr)
    pitch_mean = np.nan_to_num(pitch).mean()
    return np.concatenate([mfcc, [zcr, rms, pitch_mean]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-csv", required=True)
    ap.add_argument("--audio-col", default="path")
    ap.add_argument("--label-col", default="emotion")
    ap.add_argument("--out", default="data/baseline_report.json")
    ap.add_argument("--model", choices=["logreg", "rf"], default="logreg")
    args = ap.parse_args()

    df = pd.read_csv(args.data_csv)
    print(f"Extracting features for {len(df)} clips...")
    X = np.stack([extract_features(p) for p in df[args.audio_col]])
    y = df[args.label_col].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000) if args.model == "logreg" else RandomForestClassifier(n_estimators=200)
    clf.fit(X_train, y_train)

    report = classification_report(y_test, clf.predict(X_test), output_dict=True)
    print(classification_report(y_test, clf.predict(X_test)))

    with open(args.out, "w") as f:
        json.dump({"model": args.model, "report": report}, f, indent=2)
    print(f"Saved report to {args.out}. Compare accuracy/F1 here against WavLM eval from scripts/eval_report.py.")


if __name__ == "__main__":
    main()
