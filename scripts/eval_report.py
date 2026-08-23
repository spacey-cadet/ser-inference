"""
Track 1.4 — imbalance-aware evaluation.

Aggregate accuracy flatters the easy classes and hides exactly the failure
mode (e.g. Sad/Neutral confusion) most likely to matter in a chat context.
This script reports PR-AUC and per-class F1 alongside accuracy, and should
be what you lead with in any eval report — not the headline accuracy number.

CHANGE (additive only — every existing key is unchanged, so existing
consumers like ci-eval-gate.yml keep working untouched):
  - added per_class_recall: recall was previously not reported at all,
    only F1 and PR-AUC. Recall specifically answers "how often do we
    MISS this class" — the more actionable number for exactly the
    silent-confusion failure mode this script's docstring calls out.
  - added macro_f1 / macro_recall / macro_pr_auc: single canonical
    aggregates, computed once here rather than downstream consumers
    (e.g. scripts/promotion_gate.py) each re-deriving "mean across 8
    classes" independently and risking divergence.

Usage:
    python scripts/eval_report.py --predictions data/val_predictions.csv \
        --out data/eval_report.json
"""
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, recall_score

IDX_TO_EMOTION = {
    0: "angry", 1: "calm", 2: "disgust", 3: "fearful",
    4: "happy", 5: "neutral", 6: "sad", 7: "surprised",
}
NUM_CLASSES = 8


def _macro(per_class: dict, support: dict) -> float:
    """Mean over classes with nonzero support only — a class with 0
    examples in the held-out set has a degenerate 0/0 score that would
    silently drag the average down for no real reason."""
    values = [v for c, v in per_class.items() if support.get(c, 0) > 0]
    return float(np.mean(values)) if values else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, help="CSV with true_label, prob_0..prob_7 columns")
    ap.add_argument("--out", default="data/eval_report.json")
    ap.add_argument("--corpus", default="unspecified", help="Label this eval run, e.g. RAVDESS / cross-corpus / chat-holdout")
    args = ap.parse_args()

    df = pd.read_csv(args.predictions)
    labels = df["true_label"].values
    probs = df[[f"prob_{i}" for i in range(NUM_CLASSES)]].values
    preds = probs.argmax(axis=1)

    accuracy = accuracy_score(labels, preds)
    per_class_f1 = f1_score(labels, preds, average=None, labels=list(range(NUM_CLASSES)))
    per_class_recall = recall_score(labels, preds, average=None, labels=list(range(NUM_CLASSES)))

    per_class_pr_auc = {}
    for c in range(NUM_CLASSES):
        y_bin = (labels == c).astype(int)
        if y_bin.sum() == 0:
            per_class_pr_auc[IDX_TO_EMOTION[c]] = None
            continue
        per_class_pr_auc[IDX_TO_EMOTION[c]] = float(average_precision_score(y_bin, probs[:, c]))

    class_support = {IDX_TO_EMOTION[c]: int((labels == c).sum()) for c in range(NUM_CLASSES)}
    f1_dict = {IDX_TO_EMOTION[c]: float(per_class_f1[c]) for c in range(NUM_CLASSES)}
    recall_dict = {IDX_TO_EMOTION[c]: float(per_class_recall[c]) for c in range(NUM_CLASSES)}
    pr_auc_for_macro = {k: v for k, v in per_class_pr_auc.items() if v is not None}

    report = {
        "corpus": args.corpus,
        "n_samples": len(df),
        "accuracy": float(accuracy),
        "per_class_f1": f1_dict,
        "per_class_recall": recall_dict,
        "per_class_pr_auc": per_class_pr_auc,
        "class_support": class_support,
        "macro_f1": _macro(f1_dict, class_support),
        "macro_recall": _macro(recall_dict, class_support),
        "macro_pr_auc": _macro(pr_auc_for_macro, class_support),
        "note": "Lead with per_class_f1 / per_class_recall / per_class_pr_auc, not accuracy. "
                "Low-support / low-F1 classes (e.g. sad/neutral) are the ones "
                "most likely to matter for a chat product. macro_* fields exclude "
                "zero-support classes from the average.",
    }

    print(json.dumps(report, indent=2))
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()