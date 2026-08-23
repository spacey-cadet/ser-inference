"""
DynamoDB-backed human review queue for AWS/serverless deployments.
"""
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import boto3
from boto3.dynamodb.conditions import Key


def _to_float(value):
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, list):
        return [_to_float(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_float(v) for k, v in value.items()}
    return value


class DynamoDBReviewQueue:
    def __init__(self, table_name: str, ttl_days: int | None = 180):
        if not table_name:
            raise ValueError("REVIEW_QUEUE_TABLE must be set when STORAGE_BACKEND=aws")
        self.table = boto3.resource("dynamodb").Table(table_name)
        self.ttl_days = ttl_days

    def enqueue(
        self,
        request_id: str,
        s3_audio_key: str,
        prediction: dict,
        confidence: float,
    ) -> None:
        now = datetime.now(timezone.utc)
        item = {
            "request_id": request_id,
            "status": "pending",
            "created_at": now.isoformat(),
            "s3_audio_key": s3_audio_key,
            "prediction": _to_decimal(prediction),
            "confidence": Decimal(str(confidence)),
            "label": None,
        }
        if self.ttl_days:
            item["expires_at"] = int((now + timedelta(days=self.ttl_days)).timestamp())
        self.table.put_item(Item=item)

    def list_pending(self, limit: int = 20) -> list[dict]:
        resp = self.table.query(
            IndexName="status-created_at-index",
            KeyConditionExpression=Key("status").eq("pending"),
            Limit=limit,
            ScanIndexForward=True,
        )
        return [_to_float(item) for item in resp.get("Items", [])]

    def list_labeled_since(self, since_iso: str) -> list[dict]:
        resp = self.table.query(
            IndexName="status-created_at-index",
            KeyConditionExpression=Key("status").eq("labeled") & Key("created_at").gt(since_iso),
        )
        return [_to_float(item) for item in resp.get("Items", [])]

    def mark_labeled(self, request_id: str, label: str) -> None:
        self.table.update_item(
            Key={"request_id": request_id},
            UpdateExpression="SET #s = :labeled, #l = :label, labeled_at = :ts REMOVE expires_at",
            ExpressionAttributeNames={"#s": "status", "#l": "label"},
            ExpressionAttributeValues={
                ":labeled": "labeled",
                ":label": label,
                ":ts": datetime.now(timezone.utc).isoformat(),
            },
        )

    def mark_deployed(self, request_ids: list[str]) -> None:
        for request_id in request_ids:
            self.table.update_item(
                Key={"request_id": request_id},
                UpdateExpression="SET #s = :deployed",
                ExpressionAttributeNames={"#s": "status"},
                ExpressionAttributeValues={":deployed": "deployed"},
            )


def _to_decimal(value):
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, list):
        return [_to_decimal(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_decimal(v) for k, v in value.items()}
    return value
