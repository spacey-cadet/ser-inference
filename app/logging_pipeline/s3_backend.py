"""
S3-backed feature logging for AWS/serverless deployments.

The public interface matches SqliteFeatureLog.log_request so call sites can
select the backend through config without changing inference code.
"""
import json
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

import boto3


def _json_default(value):
    if isinstance(value, Decimal):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class S3FeatureLog:
    def __init__(self, bucket: str, feature_prefix: str = "feature-logs/", audio_prefix: str = "audio-samples/"):
        if not bucket:
            raise ValueError("AUDIO_BUCKET must be set when STORAGE_BACKEND=aws")
        self.bucket = bucket
        self.feature_prefix = feature_prefix.rstrip("/") + "/"
        self.audio_prefix = audio_prefix.rstrip("/") + "/"
        self._s3 = boto3.client("s3")

    def log_request(
        self,
        request_id: str,
        session_id: Optional[str],
        features: dict,
        top_emotion: str,
        top_score: float,
        cascade_tier: str,
        latency_ms: float,
        model_revision: str,
        consent_granted: bool,
    ):
        now = datetime.now(timezone.utc)
        row = {
            "request_id": request_id,
            "ts": time.time(),
            "logged_at": now.isoformat(),
            "session_id": session_id,
            **features,
            "top_emotion": top_emotion,
            "top_score": top_score,
            "cascade_tier": cascade_tier,
            "latency_ms": latency_ms,
            "model_revision": model_revision,
            "consent_granted": consent_granted,
        }
        key = f"{self.feature_prefix}{now:%Y-%m-%d}/{now:%H%M%S}-{request_id}.json"
        self._s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(row, default=_json_default).encode("utf-8"),
            ContentType="application/json",
        )

    def put_audio_sample(self, request_id: str, audio_bytes: bytes, content_type: str = "audio/wav") -> str:
        now = datetime.now(timezone.utc)
        key = f"{self.audio_prefix}{now:%Y-%m-%d}/{request_id}.wav"
        self._s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=audio_bytes,
            ContentType=content_type,
        )
        return key
