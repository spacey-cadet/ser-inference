"""
Track 1.2 — fit calibration on a held-out validation fold.

Takes raw softmax outputs + true labels from your validation set, builds a
reliability diagram, fits Platt scaling or isotonic regression per class, and
writes data/calibration/calibration.json for app/calibration/calibrate.py to
load at inference time. Also writes the pre/post reliability diagram PNGs as
the evidence artifact called for in the guide.

Usage:
    python scripts/calibration_fit.py \
        --predictions data/val_predictions.csv \
        --method isotonic \
        --out data/calibration/calibration.json

Expects val_predictions.csv with columns: true_label (0-7 index),
prob_0..prob_7 (raw softmax scores per class).
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

NUM_CLASSES = 8


def fit_platt(probs: np.ndarray, labels: np.ndarray) -> dict:
    params = {}
    for c in range(NUM_CLASSES):
        y_bin = (labels == c).astype(int)
        x = probs[:, c].reshape(-1, 1)
        lr = LogisticRegression()
        lr.fit(np.log(x / (1 - x + 1e-9) + 1e-9), y_bin)
        params[str(c)] = [float(lr.coef_[0][0]), float(lr.intercept_[0])]
    return params


def fit_isotonic(probs: np.ndarray, labels: np.ndarray) -> dict:
    points = {}
    for c in range(NUM_CLASSES):
        y_bin = (labels == c).astype(int)
        x = probs[:, c]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(x, y_bin)
        # Store a sampled set of (x, y) breakpoints for interpolation at inference.
        xs = np.linspace(0, 1, 101)
        ys = iso.predict(xs)
        points[str(c)] = list(zip(xs.tolist(), ys.tolist()))
    return points


def plot_reliability(probs: np.ndarray, labels: np.ndarray, title: str, out_path: str):
    top_class = probs.argmax(axis=1)
    top_conf = probs.max(axis=1)
    correct = (top_class == labels).astype(int)
    frac_pos, mean_pred = calibration_curve(correct, top_conf, n_bins=10, strategy="uniform")
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", label="perfect calibration")
    plt.plot(mean_pred, frac_pos, marker="o", label="model")
    plt.xlabel("Mean predicted confidence")
    plt.ylabel("Fraction correct")
    plt.title(title)
    plt.legend()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--method", choices=["platt", "isotonic"], default="isotonic")
    ap.add_argument("--out", default="data/calibration/calibration.json")
    args = ap.parse_args()

    df = pd.read_csv(args.predictions)
    labels = df["true_label"].values
    probs = df[[f"prob_{i}" for i in range(NUM_CLASSES)]].values

    plot_reliability(probs, labels, "Pre-calibration reliability", "data/calibration/reliability_pre.png")

    if args.method == "platt":
        params = {"method": "platt", "platt_params": fit_platt(probs, labels)}
    else:
        params = {"method": "isotonic", "isotonic_points": fit_isotonic(probs, labels)}

    with open(args.out, "w") as f:
        json.dump(params, f)
    print(f"Wrote calibration mapping ({args.method}) to {args.out}")

    # Re-plot post-calibration using the same mapping the app applies.
    from app.calibration.calibrate import CalibrationMap
    cal = CalibrationMap.load(args.out)
    calibrated = np.array([cal.apply(row.tolist()) for row in probs])
    plot_reliability(calibrated, labels, "Post-calibration reliability", "data/calibration/reliability_post.png")
    print("Wrote data/calibration/reliability_pre.png and reliability_post.png")


if __name__ == "__main__":
    main()
