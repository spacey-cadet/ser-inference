#!/usr/bin/env python
"""
scripts/promotion_gate.py — Track 1.5 decision layer.

Reads eval_report.py's canonical macro_f1 and per_class_recall fields
directly — no longer recomputes macro F1 itself, now that eval_report.py
computes it once, upstream, consistently for every caller.

Policy:
  1. macro_f1 must improve by >= --epsilon.
  2. No individual class's RECALL may regress below
     (champion_recall - --per-class-tolerance) — gates on recall, not
     F1, per-class: recall is "how often do we miss this class
     entirely," the more actionable failure mode for a chat product.
     Skipped for any class with zero support in the eval set (0/0 is
     not a real signal).
  3. (optional) --min-support: flag classes below this support;
     --strict-support makes it a hard failure instead of informational.

champion and challenger must be scored against the SAME pinned
held-out set, or class_support (and therefore this whole comparison)
won't line up.

Exit code 0 = promote, 1 = reject.

Usage:
    python scripts/promotion_gate.py \
        --champion-report champion_eval_report.json \
        --challenger-report challenger_eval_report.json \
        --epsilon 0.01 --per-class-tolerance 0.02 \
        --out decision.json
"""
import argparse
import json
import sys
from pathlib import Path

EMOTIONS = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]


def evaluate_policy(
    champion: dict,
    challenger: dict,
    epsilon: float,
    per_class_tolerance: float,
    min_support,
    strict_support: bool,
) -> dict:
    reasons = []
    checks = {}

    champ_macro = champion["macro_f1"]
    chal_macro = challenger["macro_f1"]
    delta = chal_macro - champ_macro
    primary_ok = delta >= epsilon
    checks["macro_f1"] = {
        "champion": champ_macro,
        "challenger": chal_macro,
        "delta": delta,
        "required_delta": epsilon,
        "passed": primary_ok,
    }
    if not primary_ok:
        reasons.append(
            f"macro F1 improved by {delta:.4f}, below required epsilon {epsilon:.4f} "
            f"(champion {champ_macro:.4f} -> challenger {chal_macro:.4f})"
        )

    champ_recall = champion["per_class_recall"]
    chal_recall = challenger["per_class_recall"]
    support = challenger.get("class_support", {})

    per_class_failures = []
    skipped_zero_support = []
    for cls in EMOTIONS:
        if cls not in champ_recall or cls not in chal_recall:
            per_class_failures.append(f"{cls}: missing from one of the reports")
            continue
        cls_support = support.get(cls, 0)
        if cls_support == 0:
            skipped_zero_support.append(cls)
            continue
        floor = champ_recall[cls] - per_class_tolerance
        if chal_recall[cls] < floor:
            per_class_failures.append(
                f"{cls}: recall {chal_recall[cls]:.4f} < floor {floor:.4f} "
                f"(champion was {champ_recall[cls]:.4f}, support={cls_support})"
            )
    per_class_ok = len(per_class_failures) == 0
    checks["per_class_regression"] = {
        "tolerance": per_class_tolerance,
        "failures": per_class_failures,
        "skipped_zero_support_classes": skipped_zero_support,
        "passed": per_class_ok,
    }
    if not per_class_ok:
        reasons.append("per-class recall regression: " + "; ".join(per_class_failures))

    support_ok = True
    low_support = {}
    if min_support is not None:
        low_support = {c: n for c, n in support.items() if n < min_support}
        if low_support and strict_support:
            support_ok = False
            reasons.append(
                f"held-out set has classes below min_support={min_support}: "
                f"{low_support} (strict mode — treating as reject)"
            )
    checks["min_support"] = {
        "min_support": min_support,
        "low_support_classes": low_support,
        "strict": strict_support,
        "passed": support_ok,
    }

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
    p.add_argument("--epsilon", type=float, default=0.01)
    p.add_argument("--per-class-tolerance", type=float, default=0.02)
    p.add_argument("--min-support", type=int, default=None)
    p.add_argument("--strict-support", action="store_true")
    args = p.parse_args()

    champion = json.loads(args.champion_report.read_text())
    challenger = json.loads(args.challenger_report.read_text())

    result = evaluate_policy(
        champion=champion,
        challenger=challenger,
        epsilon=args.epsilon,
        per_class_tolerance=args.per_class_tolerance,
        min_support=args.min_support,
        strict_support=args.strict_support,
    )

    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    sys.exit(0 if result["decision"] == "promote" else 1)


if __name__ == "__main__":
    main()