"""
End-to-end predict pipeline for one audio clip.

decode -> mono/16k -> validate (duration/RMS) -> VAD trim -> WavLM forward ->
pooling -> classifier -> softmax -> calibration -> confidence cascade

This is the function both the champion and challenger paths call — canary
routing (app/routing/canary.py) just decides which LoadedModel gets passed
in.
"""
import time
from dataclasses import dataclass

import torch

from app.audio.decode import decode_audio_to_tensor, to_mono_16k
from app.audio.validate import AudioRejected, validate_and_extract_features, AudioFeatures
from app.audio.vad import trim_silence
from app.calibration.calibrate import CalibrationMap
from app.inference.cascade import apply_cascade, CascadeResult
from app.model.classifier import IDX_TO_EMOTION, NUM_CLASSES
from app.model.loader import LoadedModel


@dataclass
class PredictOutcome:
    cascade: CascadeResult
    features: AudioFeatures
    latency_ms: float
    model_revision: str


def run_inference(model: LoadedModel, waveform: torch.Tensor) -> list[dict]:
    """Raw forward pass -> ranked emotion list (no calibration/cascade applied yet)."""
    wav_lens = torch.tensor([1.0])
    with torch.no_grad():
        features = model.wavlm(waveform)
        if isinstance(features, dict):
            features = features["last_hidden_state"]
        pooled = model.pooling(features, wav_lens)
        logits = model.classifier(pooled)
        probs = torch.softmax(logits, dim=-1).squeeze(0)

    probs_list = probs.tolist()
    ranked = sorted(
        [{"emotion": IDX_TO_EMOTION[i], "score": round(probs_list[i], 4)} for i in range(NUM_CLASSES)],
        key=lambda x: x["score"],
        reverse=True,
    )
    return ranked


def predict_from_path(
    path: str,
    model: LoadedModel,
    calibration: CalibrationMap,
    settings,
) -> PredictOutcome:
    start = time.perf_counter()

    signal, sample_rate = decode_audio_to_tensor(path)
    signal = to_mono_16k(signal, sample_rate)

    # Validate on the resampled 16kHz signal so duration/RMS thresholds are
    # consistent regardless of source codec.
    features = validate_and_extract_features(
        signal,
        sample_rate=16000,
        min_duration_sec=settings.min_duration_sec,
        max_duration_sec=settings.max_duration_sec,
        rms_silence_floor=settings.rms_silence_floor,
    )  # raises AudioRejected on failure — let the caller handle it

    if settings.vad_enabled:
        signal = trim_silence(signal, sample_rate=16000)

    waveform = signal.squeeze(0).unsqueeze(0)  # [1, time]
    ranking = run_inference(model, waveform)

    calibrated_scores = calibration.apply([r["score"] for r in ranking])
    for r, cs in zip(ranking, calibrated_scores):
        r["score"] = round(cs, 4)
    ranking = sorted(ranking, key=lambda x: x["score"], reverse=True)

    cascade = apply_cascade(ranking, settings.threshold_low, settings.threshold_high)

    latency_ms = (time.perf_counter() - start) * 1000
    return PredictOutcome(
        cascade=cascade,
        features=features,
        latency_ms=latency_ms,
        model_revision=model.revision_tag,
    )
