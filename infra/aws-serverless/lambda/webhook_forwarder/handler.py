"""
Tiny SNS -> Slack/Discord webhook forwarder. Exists only so CloudWatch
alarms can reuse alerts/webhook.py's existing channel instead of AWS
SNS email/SMS (which is an extra thing to check instead of the one
place you already look for retrain/drift alerts).
"""
import json
import os
import urllib.request

WEBHOOK_URL = os.environ["ALERT_WEBHOOK_URL"]


def handler(event, context):
    for record in event.get("Records", []):
        sns_message = record["Sns"]["Message"]
        try:
            payload = json.loads(sns_message)
            alarm_name = payload.get("AlarmName", "unknown alarm")
            reason = payload.get("NewStateReason", "")
            text = f":rotating_light: CloudWatch alarm **{alarm_name}** fired: {reason}"
        except (json.JSONDecodeError, TypeError):
            text = f":rotating_light: CloudWatch alarm fired: {sns_message}"

        body = json.dumps({"content": text, "text": text}).encode("utf-8")
        req = urllib.request.Request(
            WEBHOOK_URL, data=body, headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req, timeout=5)

    return {"statusCode": 200}
