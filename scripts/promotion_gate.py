#!/usr/bin/env python
"""
scripts/promotion_gate.py — Track 1.5 decision layer.

eval_report.py reports; it doesn't decide. This script is the missing
piece: it takes two eval_report.py outputs (champion, challenger — both
scored against the SAME pinned held-out set, or the comparison is
meaningless) and applies an explicit promotion policy.

Policy, in order:
  1. Primary metric (default: macro F1) must improve by >= --epsilon.
     Guards against promoting on noise from a small new batch.
  2. No individual class's recall may regress below
     (champion_recall - --per-class-tolerance).
     Guards against trading overall gain for silently breaking one
     emotion class — a real risk when only the head is retrained on a
     small incremental batch.
  3. (optional) --min-support: challenger's held-out set must have at
     least this many examples in every class, so a class with 3 samples
     doesn't swing the decision.

Exit code 0 = promote, 1 = reject. Both cases print/write a decision
JSON with the specific reason, so alerts/webhook.py has something
concrete to post instead of just "retrain finished."

Usage:
    python scripts/promotion_gate.py \
        --champion-report champion_eval_report.json \
        --challenger-report challenger_eval_report.json \
        --out decision.json

This assumes eval_report.py's JSON has roughly this shape — ADJUST THE
KEY NAMES in `_get` calls below to match your actual output before
relying on this:

{
  "accuracy": 0.81,
  "f1": {"macro": 0.74, "per_class": {"Sad": 0.61, "Neutral": 0.70, ...}},
  "pr_auc": {"macro": 0.79, "per_class": {"Sad": 0.68, ...}},
  "recall": {"macro": 0.73, "per_class": {"Sad": 0.55, "Neutral": 0.66, ...}},
  "support": {"Sad": 120, "Neutral": 340, ...}
}
"""
import argparse
import json
import sys
from pathlib import Path


def _get(report: dict, dotted_path: str):
    """Fetch a nested key like 'f1.macro' or 'recall.per_class'. Raises
    a clear error rather than a bare KeyError if the schema doesn't match —
    check your actual eval_report.py output and adjust the paths passed
    on the CLI (--primary-metric, etc.) if this fires."""
    node = report
    for part in dotted_path.split("."):
        if part not in node:
            raise KeyError(
                f"Expected key '{part}' (from '{dotted_path}') not found "
                f"in report. Available keys at this level: {list(node.keys())}. "
                f"Check eval_report.py's actual JSON schema and adjust "
                f"--primary-metric / --recall-key accordingly."
            )
        node = node[part]
    return node


def evaluate_policy(
    champion: dict,
    challenger: dict,
    primary_metric: str,
    epsilon: float,
    recall_key: str,
    per_class_tolerance: float,
    support_key: str,
    min_support: int,
) -> dict:
    reasons = []
    checks = {}

    # --- Check 1: primary metric must improve by at least epsilon ---
    champ_primary = _get(champion, primary_metric)
    chal_primary = _get(challenger, primary_metric)
    delta = chal_primary - champ_primary
    primary_ok = delta >= epsilon
    checks["primary_metric"] = {
        "metric": primary_metric,
        "champion": champ_primary,
        "challenger": chal_primary,
        "delta": delta,
        "required_delta": epsilon,
        "passed": primary_ok,
    }
    if not primary_ok:
        reasons.append(
            f"primary metric '{primary_metric}' improved by {delta:.4f}, "
            f"below required epsilon {epsilon:.4f}"
        )

    # --- Check 2: no per-class recall regression past tolerance ---
    champ_recall = _get(champion, recall_key)
    chal_recall = _get(challenger, recall_key)
    per_class_failures = []
    for cls, champ_val in champ_recall.items():
        chal_val = chal_recall.get(cls)
        if chal_val is None:
            per_class_failures.append(f"{cls}: missing from challenger report")
            continue
        floor = champ_val - per_class_tolerance
        if chal_val < floor:
            per_class_failures.append(
                f"{cls}: recall {chal_val:.4f} < floor {floor:.4f} "
                f"(champion was {champ_val:.4f})"
            )
    per_class_ok = len(per_class_failures) == 0
    checks["per_class_regression"] = {
        "tolerance": per_class_tolerance,
        "failures": per_class_failures,
        "passed": per_class_ok,
    }
    if not per_class_ok:
        reasons.append(
            "per-class recall regression: " + "; ".join(per_class_failures)
        )

    # --- Check 3 (optional): minimum support per class in eval set ---
    support_ok = True
    if min_support is not None:
        support = _get(challenger, support_key)
        low_support = {c: n for c, n in support.items() if n < min_support}
        support_ok = len(low_support) == 0
        checks["min_support"] = {
            "min_support": min_support,
            "low_support_classes": low_support,
            "passed": support_ok,
        }
        if not support_ok:
            reasons.append(
                f"held-out set has classes below min_support={min_support}: "
                f"{low_support} — decision is unreliable, treating as reject"
            )

    promote = primary_ok and per_class_ok and support_ok
    return {
        "decision": "promote" if promote else "reject",
        "reasons": reasons if reasons else ["all checks passed"],
        "checks": checks,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--champion-report", required=True, type=Path)
    p.add_argument("--challenger-report", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--primary-metric", default="f1.macro",
                    help="Dotted path into the report JSON, e.g. 'f1.macro' or 'pr_auc.macro'")
    p.add_argument("--epsilon", type=float, default=0.01,
                    help="Minimum required improvement in the primary metric to promote")
    p.add_argument("--recall-key", default="recall.per_class",
                    help="Dotted path to the per-class recall dict")
    p.add_argument("--per-class-tolerance", type=float, default=0.02,
                    help="Max allowed recall regression on any single class")
    p.add_argument("--support-key", default="support",
                    help="Dotted path to the per-class support-count dict")
    p.add_argument("--min-support", type=int, default=None,
                    help="If set, reject when any class has fewer than this many held-out examples")
    args = p.parse_args()

    champion = json.loads(args.champion_report.read_text())
    challenger = json.loads(args.challenger_report.read_text())

    result = evaluate_policy(
        champion=champion,
        challenger=challenger,
        primary_metric=args.primary_metric,
        epsilon=args.epsilon,
        recall_key=args.recall_key,
        per_class_tolerance=args.per_class_tolerance,
        support_key=args.support_key,
        min_support=args.min_support,
    )

    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    sys.exit(0 if result["decision"] == "promote" else 1)


if __name__ == "__main__":
    main()
