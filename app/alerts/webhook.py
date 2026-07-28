"""
Slack / Discord incoming webhook alerting (Track 2.5).

Deliberately minimal — a two-line requests.post() call, triggered on:
  - KS-test p-value < DRIFT_P_VALUE_THRESHOLD (from scripts/drift_check.py)
  - low-confidence rate crossing LOW_CONFIDENCE_RATE_THRESHOLD in a rolling
    window (checked in-process, see app/api/routes.py)

This notifies; it does not page or escalate. See docs/ROLLOUT.md for what a
paid on-call/paging system would add on top.
"""
import requests


def send_alert(webhook_url: str, message: str):
    if not webhook_url:
        print(f"[ALERT - no webhook configured] {message}")
        return
    try:
        # Slack and Discord both accept a simple {"text": ...} / {"content": ...}
        # payload for basic incoming webhooks — try Slack's key first, Discord
        # ignores unknown keys and reads "content".
        requests.post(webhook_url, json={"text": message, "content": message}, timeout=5)
    except Exception as e:  # noqa: BLE001 - alerting must never crash the request path
        print(f"Alert webhook failed: {e}")
