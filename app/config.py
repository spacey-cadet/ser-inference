"""
Central configuration for the SER production pipeline.

Everything tunable lives here and is sourced from environment variables so the
same image can run in dev / staging / production Spaces with different
behavior purely via env vars (no code changes, no rebuilds).
"""
import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Model ────────────────────────────────────────────────────────────
    model_id: str = os.environ.get("MODEL_ID", "space-cadet/wavlm-ser")
    hf_token: str = os.environ.get("HF_TOKEN", "")
    cache_dir: str = os.environ.get("CACHE_DIR", "./model_cache")

    # Champion is always loaded. Challenger is optional (Track 3 canary).
    challenger_model_id: str = os.environ.get("CHALLENGER_MODEL_ID", "")
    challenger_revision: str = os.environ.get("CHALLENGER_REVISION", "main")
    canary_pct: float = float(os.environ.get("CANARY_PCT", "0"))  # 0-100

    # ── Calibration & thresholding (Track 1.2 / 1.3) ───────────────────
    calibration_path: str = os.environ.get("CALIBRATION_PATH", "./data/calibration/calibration.json")
    threshold_low: float = float(os.environ.get("THRESHOLD_LOW", "0.40"))
    threshold_high: float = float(os.environ.get("THRESHOLD_HIGH", "0.65"))

    # ── Audio validation (Track 2.1) ────────────────────────────────────
    min_duration_sec: float = float(os.environ.get("MIN_DURATION_SEC", "0.5"))
    max_duration_sec: float = float(os.environ.get("MAX_DURATION_SEC", "30.0"))
    rms_silence_floor: float = float(os.environ.get("RMS_SILENCE_FLOOR", "0.001"))
    vad_enabled: bool = os.environ.get("VAD_ENABLED", "true").lower() == "true"

    # ── Feature / drift logging (Track 2.2) ─────────────────────────────
    feature_log_backend: str = os.environ.get("FEATURE_LOG_BACKEND", "sqlite")  # sqlite | hf_dataset | none
    sqlite_path: str = os.environ.get("SQLITE_PATH", "./data/telemetry.db")
    hf_dataset_repo: str = os.environ.get("HF_DATASET_REPO", "")
    drift_reference_path: str = os.environ.get("DRIFT_REFERENCE_PATH", "./data/drift_reference/train_features.json")

    # ── Consent / retention (Track 2.4) ─────────────────────────────────
    consent_required: bool = os.environ.get("CONSENT_REQUIRED", "true").lower() == "true"
    retention_days: int = int(os.environ.get("RETENTION_DAYS", "30"))

    # ── Session state (Track 3.3) ───────────────────────────────────────
    session_backend: str = os.environ.get("SESSION_BACKEND", "memory")  # memory | redis
    redis_url: str = os.environ.get("REDIS_URL", "")
    session_decay: float = float(os.environ.get("SESSION_DECAY", "0.6"))  # weight on new obs
    session_history_len: int = int(os.environ.get("SESSION_HISTORY_LEN", "10"))

    # ── Alerting (Track 2.5) ─────────────────────────────────────────────
    alert_webhook_url: str = os.environ.get("ALERT_WEBHOOK_URL", "")
    drift_p_value_threshold: float = float(os.environ.get("DRIFT_P_VALUE_THRESHOLD", "0.05"))
    low_confidence_rate_threshold: float = float(os.environ.get("LOW_CONFIDENCE_RATE_THRESHOLD", "0.30"))

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
