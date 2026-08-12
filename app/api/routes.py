import os
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.audio.validate import AudioRejected
from app.inference.predict import predict_from_path
from app.logging_pipeline.consent import evaluate_consent, strip_identifying_metadata
from app.logging_pipeline.review_queue import export_for_review, queue_stats
from app.routing.canary import select_model

router = APIRouter()


@router.get("/")
def health(request: Request):
    state = request.app.state
    return {
        "status": "ok",
        "champion_revision": state.registry.champion.revision_tag,
        "challenger_revision": (state.registry.challenger.revision_tag if state.registry.challenger else None),
        "calibrated": state.calibration.method != "none",
    }


@router.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    session_id: str | None = Form(default=None),
    consent: bool | None = Form(default=None),
):
    """
    Main inference endpoint.

    Form fields beyond the audio file:
      - session_id: opaque client-generated ID, used for session-level
        rolling emotion state (Track 3.3). Optional — if absent, no session
        state is updated.
      - consent: whether this request's features may be logged for drift
        monitoring / the review queue (Track 2.4). Required if
        CONSENT_REQUIRED=true.
    """
    state = request.app.state
    settings = state.settings
    request_id = str(uuid.uuid4())

    suffix = Path(file.filename).suffix if file.filename else ".audio"
    audio_bytes = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        path = tmp.name

    try:
        model, is_challenger = select_model(state.registry, settings.canary_pct)

        try:
            outcome = predict_from_path(path, model, state.calibration, settings)
        except AudioRejected as e:
            # Chat-safe default: don't error the request, return a no-signal
            # response the calling app can treat as "no adjustment."
            return {
                "request_id": request_id,
                "status": "rejected",
                "reason": e.reason,
                "response_emotion": "neutral",
            }

        # Session state update (Track 3.3)
        session_emotion = None
        if session_id:
            session_id = strip_identifying_metadata(session_id)
            session_state = state.session_store.update(
                session_id=session_id,
                emotion=outcome.cascade.response_emotion,
                score=outcome.cascade.top_score,
                decay=settings.session_decay,
                history_len=settings.session_history_len,
            )
            session_emotion = session_state.dominant_emotion

        # Feature/drift logging + review queue (Track 2.2 / 2.3), gated on consent (Track 2.4)
        consent_decision = evaluate_consent(consent, settings.consent_required)
        if consent_decision.can_log and state.feature_log is not None:
            state.feature_log.log_request(
                request_id=request_id,
                session_id=session_id,
                features=outcome.features.to_dict(),
                top_emotion=outcome.cascade.top_emotion,
                top_score=outcome.cascade.top_score,
                cascade_tier=outcome.cascade.tier.value,
                latency_ms=outcome.latency_ms,
                model_revision=outcome.model_revision,
                consent_granted=True,
            )
            should_enqueue_review = (
                outcome.cascade.log_for_review
                and state.review_queue is not None
                and hasattr(state.feature_log, "put_audio_sample")
            )
            if should_enqueue_review:
                s3_audio_key = state.feature_log.put_audio_sample(
                    request_id=request_id,
                    audio_bytes=audio_bytes,
                    content_type=file.content_type or "audio/wav",
                )
                state.review_queue.enqueue(
                    request_id=request_id,
                    s3_audio_key=s3_audio_key,
                    prediction={
                        "label": outcome.cascade.top_emotion,
                        "response_emotion": outcome.cascade.response_emotion,
                        "scores": outcome.cascade.ranking,
                    },
                    confidence=outcome.cascade.top_score,
                )

        return {
            "request_id": request_id,
            "status": "ok",
            "top_emotion": outcome.cascade.top_emotion,
            "response_emotion": outcome.cascade.response_emotion,
            "cascade_tier": outcome.cascade.tier.value,
            "ranking": outcome.cascade.ranking,
            "session_dominant_emotion": session_emotion,
            "latency_ms": round(outcome.latency_ms, 1),
            "model_revision": outcome.model_revision,
            "is_challenger": is_challenger,
            "logged": consent_decision.can_log and state.feature_log is not None,
        }
    finally:
        if os.path.exists(path):
            os.remove(path)


@router.get("/admin/review-queue/stats")
def review_queue_stats(request: Request):
    """Track 2.3 — queue depth, for dashboards/alerting."""
    state = request.app.state
    if state.review_queue is None:
        raise HTTPException(status_code=404, detail="review queue disabled")
    stats = queue_stats(state.review_queue)
    return {"unreviewed_count": stats.unreviewed_count, "oldest_unreviewed_ts": stats.oldest_unreviewed_ts}


@router.get("/admin/review-queue/export")
def review_queue_export(request: Request, limit: int = 100):
    """Track 2.3 — pull a batch for manual labeling."""
    state = request.app.state
    if state.review_queue is None:
        raise HTTPException(status_code=404, detail="review queue disabled")
    return {"rows": export_for_review(state.review_queue, limit=limit)}
