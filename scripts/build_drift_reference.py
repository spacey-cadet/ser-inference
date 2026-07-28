"""
Builds the training-distribution reference file that scripts/drift_check.py
compares live traffic against. Run this once per training set / after each
retrain — not continuously.

Usage:
    python scripts/build_drift_reference.py --data-csv data/labels.csv \
        --audio-col path --out data/drift_reference/train_features.json
"""
import argparse
import json

from app.audio.decode import decode_audio_to_tensor, to_mono_16k
from app.audio.validate import validate_and_extract_features
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-csv", required=True)
    ap.add_argument("--audio-col", default="path")
    ap.add_argument("--out", default="data/drift_reference/train_features.json")
    args = ap.parse_args()

    df = pd.read_csv(args.data_csv)
    out = {"duration_sec": [], "rms_energy": [], "spectral_centroid": [], "pitch_estimate": []}

    for path in df[args.audio_col]:
        try:
            signal, sr = decode_audio_to_tensor(path)
            signal = to_mono_16k(signal, sr)
            features = validate_and_extract_features(
                signal, sample_rate=16000, min_duration_sec=0.0, max_duration_sec=1e9, rms_silence_floor=0.0
            )
        except Exception as e:  # noqa: BLE001
            print(f"Skipping {path}: {e}")
            continue
        out["duration_sec"].append(features.duration_sec)
        out["rms_energy"].append(features.rms_energy)
        out["spectral_centroid"].append(features.spectral_centroid)
        out["pitch_estimate"].append(features.pitch_estimate)

    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"Wrote reference distribution ({len(df)} clips) to {args.out}")


if __name__ == "__main__":
    main()
