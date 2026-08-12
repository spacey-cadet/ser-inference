#!/usr/bin/env python
"""
scripts/check_retrain_policy.py

The cheap daily check. Decides whether the expensive Kaggle retrain job
should run today, based on:
  - volume floor: >= --min-new-labels new labeled items since the last
    successful training run
  - cooldown floor: >= --cooldown-days since the last training run
    (prevents back-to-back retrains from a labeling burst)
  - staleness ceiling: force a retrain after --max-staleness-days even
    below the volume floor (prevents a slow trickle stalling forever)
  - optional drift trigger: if a drift_check.py run in the last
    --drift-lookback-days flagged significant drift, retrain regardless
    of the volume floor (respects cooldown floor)

Writes "true"/"false" to GITHUB_OUTPUT so the calling workflow can gate
the expensive job step on it, and always writes a decision.json with
the reasoning for the alert.

Usage:
    python scripts/check_retrain_policy.py \
        --review-queue-table ser-inference-review-queue \
        --retrain-state-table ser-inference-retrain-state \
        --min-new-labels 50 --cooldown-days 3 --max-staleness-days 60
"""
import argparse
import json
import os
from datetime import datetime, timedelta, timezone

import boto3
from boto3.dynamodb.conditions import Key


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--review-queue-table", required=True)
    p.add_argument("--retrain-state-table", required=True)
    p.add_argument("--min-new-labels", type=int, default=50)
    p.add_argument("--cooldown-days", type=int, default=3)
    p.add_argument("--max-staleness-days", type=int, default=60)
    p.add_argument("--drift-flagged", action="store_true",
                    help="Pass this if a prior drift_check.py step flagged drift this run")
    p.add_argument("--out", default="retrain_decision.json")
    args = p.parse_args()

    dynamodb = boto3.resource("dynamodb")
    review_queue = dynamodb.Table(args.review_queue_table)
    retrain_state = dynamodb.Table(args.retrain_state_table)

    state_resp = retrain_state.get_item(Key={"state_key": "watermark"})
    state = state_resp.get("Item", {})
    last_trained_at = state.get("last_trained_at")
    now = datetime.now(timezone.utc)

    if last_trained_at is None:
        # never trained via this pipeline before — treat as maximally stale
        reasons = ["no previous training run recorded — treating as stale, will train if any labels exist"]
        days_since_last = None
    else:
        last_dt = datetime.fromisoformat(last_trained_at)
        days_since_last = (now - last_dt).days
        reasons = []

    resp = review_queue.query(
        IndexName="status-created_at-index",
        KeyConditionExpression=(
            Key("status").eq("labeled")
            & Key("created_at").gt(last_trained_at or "1970-01-01T00:00:00")
        ),
        Select="COUNT",
    )
    new_label_count = resp["Count"]

    cooldown_ok = days_since_last is None or days_since_last >= args.cooldown_days
    volume_met = new_label_count >= args.min_new_labels
    staleness_forced = days_since_last is not None and days_since_last >= args.max_staleness_days
    drift_forced = args.drift_flagged

    if not cooldown_ok:
        should_train = False
        reasons.append(
            f"cooldown not met: {days_since_last}d since last run, "
            f"need >= {args.cooldown_days}d"
        )
    elif volume_met:
        should_train = True
        reasons.append(f"volume floor met: {new_label_count} new labels >= {args.min_new_labels}")
    elif staleness_forced:
        should_train = True
        reasons.append(
            f"staleness ceiling forced retrain: {days_since_last}d >= "
            f"{args.max_staleness_days}d, despite only {new_label_count} new labels"
        )
    elif drift_forced:
        should_train = True
        reasons.append("drift check flagged significant drift — retraining despite volume floor")
    else:
        should_train = False
        reasons.append(
            f"volume floor not met: {new_label_count} new labels < {args.min_new_labels}, "
            f"and not stale/drifted enough to force"
        )

    decision = {
        "should_train": should_train,
        "new_label_count": new_label_count,
        "days_since_last_train": days_since_last,
        "reasons": reasons,
    }

    with open(args.out, "w") as f:
        json.dump(decision, f, indent=2)
    print(json.dumps(decision, indent=2))

    gh_output = os.environ.get("GITHUB_OUTPUT")
    if gh_output:
        with open(gh_output, "a") as f:
            f.write(f"should_train={'true' if should_train else 'false'}\n")


if __name__ == "__main__":
    main()
