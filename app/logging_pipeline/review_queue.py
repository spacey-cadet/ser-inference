"""
Label-collection / review queue (Track 2.3).

This is the mechanism the guide reframes as the core of Track 2: every
prediction the cascade marks below the "confident" tier is logged with a
review flag, and periodically sampled for manual review/labeling. When a
retrain happens, this queue is the first place to pull real chat-domain
examples from — specifically because it's the data the model already
struggled with.

This module is thin on purpose: it exposes queue depth / throughput so you
can alert on "queue growing faster than it's reviewed" (a drift-acceleration
signal in its own right), and a mark_reviewed hook for whatever manual
review UI/process you use (a spreadsheet export is a perfectly good v1).
"""
from dataclasses import dataclass
import sqlite3


@dataclass
class QueueStats:
    unreviewed_count: int
    oldest_unreviewed_ts: float | None


class FeatureLogReviewQueue:
    def __init__(self, feature_log, sqlite_path: str | None = None):
        self.feature_log = feature_log
        self.sqlite_path = sqlite_path

    def list_pending(self, limit: int = 100) -> list[dict]:
        if self.feature_log is None:
            return []
        return self.feature_log.recent_low_confidence(limit=limit)

    def mark_labeled(self, request_id: str, label: str) -> None:
        self.mark_reviewed([request_id])

    def mark_reviewed(self, request_ids: list[str]):
        if not self.sqlite_path:
            raise ValueError("mark_reviewed is only supported for the sqlite review queue")
        mark_reviewed(self.sqlite_path, request_ids)


def build_review_queue(settings, feature_log=None):
    if settings.storage_backend == "aws":
        from app.logging_pipeline.dynamodb_review_queue import DynamoDBReviewQueue
        return DynamoDBReviewQueue(settings.review_queue_table)
    if feature_log is None:
        return None
    return FeatureLogReviewQueue(feature_log, settings.sqlite_path if settings.feature_log_backend == "sqlite" else None)


def queue_stats(review_queue) -> QueueStats:
    rows = review_queue.list_pending(limit=10_000)
    if not rows:
        return QueueStats(unreviewed_count=0, oldest_unreviewed_ts=None)
    oldest = min(r.get("ts") or _parse_iso_ts(r.get("created_at")) for r in rows)
    return QueueStats(unreviewed_count=len(rows), oldest_unreviewed_ts=oldest)


def export_for_review(review_queue, limit: int = 100) -> list[dict]:
    """Pulls a batch of low-confidence rows for manual labeling. Intended to
    be exported to a spreadsheet / labeling tool, not labeled in-process."""
    return review_queue.list_pending(limit=limit)


def mark_reviewed(sqlite_path: str, request_ids: list[str]):
    conn = sqlite3.connect(sqlite_path)
    try:
        conn.executemany(
            "UPDATE requests SET reviewed = 1 WHERE request_id = ?",
            [(rid,) for rid in request_ids],
        )
        conn.commit()
    finally:
        conn.close()


def _parse_iso_ts(value: str | None) -> float:
    if not value:
        return 0.0
    from datetime import datetime

    return datetime.fromisoformat(value).timestamp()
