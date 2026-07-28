"""
Track 2.2 — serving-gate drift monitoring.

Compares the live feature distribution (logged per-request by
app/logging_pipeline/feature_log.py) against the training distribution using
scipy.stats.ks_2samp. Intended to run periodically — a scheduled GitHub
Action (see .github/workflows/drift-check.yml) or a manual notebook run.

Expect this to fire early and often at first: chat audio characteristics
diverging from RAVDESS/IEMOCAP/CREMA-D is not a bug, it's the exact gap the
cross-corpus numbers already predict.

Usage:
    python scripts/drift_check.py
"""
import json
import sys

from scipy.stats import ks_2samp

from app.alerts.webhook import send_alert
from app.config import get_settings
from app.logging_pipeline.feature_log import build_feature_log
from app.logging_pipeline.review_queue import queue_stats


def load_reference_distribution(path: str) -> dict:
    """
    Reference distribution built once (e.g. from your RAVDESS/IEMOCAP/CREMA-D
    training set) via scripts/build_drift_reference.py — see that script's
    docstring for the expected format. Kept separate from this script since
    it only needs to run once per training set, not on every drift check.
    """
    with open(path) as f:
        return json.load(f)


def main():
    settings = get_settings()
    feature_log = build_feature_log(settings)
    if feature_log is None:
        print("Feature logging disabled (FEATURE_LOG_BACKEND=none); nothing to check.")
        return

    live = feature_log.feature_distribution()
    try:
        reference = load_reference_distribution(settings.drift_reference_path)
    except FileNotFoundError:
        print(f"No reference distribution at {settings.drift_reference_path}. "
              "Run scripts/build_drift_reference.py against your training set first.")
        sys.exit(1)

    results = {}
    fired = False
    for feature_name, live_values in live.items():
        ref_values = reference.get(feature_name, [])
        if len(live_values) < 20 or len(ref_values) < 20:
            results[feature_name] = {"skipped": "insufficient samples"}
            continue
        stat, p_value = ks_2samp(live_values, ref_values)
        results[feature_name] = {"ks_statistic": float(stat), "p_value": float(p_value)}
        if p_value < settings.drift_p_value_threshold:
            fired = True

    stats = queue_stats(feature_log)
    low_conf_rate = None
    total_logged = sum(len(v) for v in live.values()) or 1
    if stats.unreviewed_count:
        low_conf_rate = stats.unreviewed_count / max(total_logged, 1)

    print(json.dumps({"drift": results, "review_queue_unreviewed": stats.unreviewed_count,
                       "low_confidence_rate": low_conf_rate}, indent=2))

    if fired:
        send_alert(
            settings.alert_webhook_url,
            f":rotating_light: SER drift check: KS-test fired (p < {settings.drift_p_value_threshold}) "
            f"on one or more features. Details: {json.dumps(results)}",
        )
    if low_conf_rate is not None and low_conf_rate > settings.low_confidence_rate_threshold:
        send_alert(
            settings.alert_webhook_url,
            f":warning: SER low-confidence rate {low_conf_rate:.1%} exceeds threshold "
            f"{settings.low_confidence_rate_threshold:.0%}. Review queue depth: {stats.unreviewed_count}.",
        )


if __name__ == "__main__":
    main()
