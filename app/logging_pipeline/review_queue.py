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


@dataclass
class QueueStats:
    unreviewed_count: int
    oldest_unreviewed_ts: float | None


def queue_stats(feature_log) -> QueueStats:
    rows = feature_log.recent_low_confidence(limit=10_000)
    if not rows:
        return QueueStats(unreviewed_count=0, oldest_unreviewed_ts=None)
    oldest = min(r["ts"] for r in rows)
    return QueueStats(unreviewed_count=len(rows), oldest_unreviewed_ts=oldest)


def export_for_review(feature_log, limit: int = 100) -> list[dict]:
    """Pulls a batch of low-confidence rows for manual labeling. Intended to
    be exported to a spreadsheet / labeling tool, not labeled in-process."""
    return feature_log.recent_low_confidence(limit=limit)


def mark_reviewed(sqlite_path: str, request_ids: list[str]):
    import sqlite3
    conn = sqlite3.connect(sqlite_path)
    try:
        conn.executemany(
            "UPDATE requests SET reviewed = 1 WHERE request_id = ?",
            [(rid,) for rid in request_ids],
        )
        conn.commit()
    finally:
        conn.close()
