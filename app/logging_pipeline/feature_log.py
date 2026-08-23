"""
Per-request feature + outcome logging (Track 2.2).

Logs a small feature vector (duration, RMS energy, spectral centroid, pitch
estimate) plus prediction outcome per inference request. This is what
scripts/drift_check.py later compares against the training distribution with
scipy.stats.ks_2samp.

Two backends:
  - sqlite: local file, fine if the Space has persistent storage.
  - hf_dataset: append-only private Hugging Face Dataset — survives Space
    restarts/redeploys, which a local SQLite file on ephemeral storage won't.

Consent (Track 2.4) is enforced at the call site (app/logging_pipeline/consent.py)
before this module is ever invoked with real audio references — this module
itself just persists whatever it's given.
"""
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    request_id TEXT PRIMARY KEY,
    ts REAL NOT NULL,
    session_id TEXT,
    duration_sec REAL,
    rms_energy REAL,
    spectral_centroid REAL,
    pitch_estimate REAL,
    top_emotion TEXT,
    top_score REAL,
    cascade_tier TEXT,
    latency_ms REAL,
    model_revision TEXT,
    consent_granted INTEGER,
    reviewed INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(ts);
CREATE INDEX IF NOT EXISTS idx_requests_reviewed ON requests(reviewed);
"""


class SqliteFeatureLog:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self._path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

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
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO requests
                (request_id, ts, session_id, duration_sec, rms_energy, spectral_centroid,
                 pitch_estimate, top_emotion, top_score, cascade_tier, latency_ms,
                 model_revision, consent_granted, reviewed)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,0)""",
                (
                    request_id, time.time(), session_id,
                    features.get("duration_sec"), features.get("rms_energy"),
                    features.get("spectral_centroid"), features.get("pitch_estimate"),
                    top_emotion, top_score, cascade_tier, latency_ms, model_revision,
                    int(consent_granted),
                ),
            )

    def recent_low_confidence(self, limit: int = 100) -> list[dict]:
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM requests WHERE cascade_tier != 'confident' "
                "AND reviewed = 0 ORDER BY ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def feature_distribution(self, since_ts: Optional[float] = None) -> dict:
        """Returns lists of feature values suitable for ks_2samp against a
        reference distribution."""
        query = "SELECT duration_sec, rms_energy, spectral_centroid, pitch_estimate FROM requests"
        params = ()
        if since_ts:
            query += " WHERE ts >= ?"
            params = (since_ts,)
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        out = {"duration_sec": [], "rms_energy": [], "spectral_centroid": [], "pitch_estimate": []}
        for r in rows:
            for i, key in enumerate(out.keys()):
                if r[i] is not None:
                    out[key].append(r[i])
        return out

    def purge_older_than(self, retention_days: int):
        """Enforces the retention window from the consent/retention policy."""
        cutoff = time.time() - retention_days * 86400
        with self._conn() as conn:
            conn.execute("DELETE FROM requests WHERE ts < ?", (cutoff,))


def build_feature_log(settings):
    if settings.storage_backend == "aws":
        from app.logging_pipeline.s3_backend import S3FeatureLog
        return S3FeatureLog(
            bucket=settings.audio_bucket,
            feature_prefix=settings.feature_log_prefix,
            audio_prefix=settings.audio_sample_prefix,
        )
    if settings.feature_log_backend == "sqlite":
        return SqliteFeatureLog(settings.sqlite_path)
    if settings.feature_log_backend == "hf_dataset":
        from app.logging_pipeline.hf_dataset_log import HFDatasetFeatureLog
        return HFDatasetFeatureLog(settings.hf_dataset_repo, settings.hf_token)
    return None  # feature_log_backend == "none": logging disabled entirely
