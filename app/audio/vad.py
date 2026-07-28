"""
Voice-activity detection (Track 2.1).

Strips silence / background noise before scoring rather than feeding dead
air through the model. Uses silero-vad (torch.hub, free, no API key) as the
primary path. If silero can't load (e.g. no network to torch.hub on a
locked-down Space), falls back to webrtcvad, which ships as a plain wheel.

Both are optional at the config level (VAD_ENABLED=false) — if a deployment
can't reach either, the pipeline still runs on the RMS-floor rejection alone.
"""
import torch

_silero_model = None
_silero_utils = None


def _load_silero():
    global _silero_model, _silero_utils
    if _silero_model is not None:
        return _silero_model, _silero_utils
    _silero_model, _silero_utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
    )
    return _silero_model, _silero_utils


def trim_silence(signal: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
    """
    signal: [1, time] float32 @ 16kHz
    Returns the same tensor with leading/trailing/interior silence removed
    based on detected speech segments. Falls back to the original signal
    (fail-open, not fail-closed — we'd rather score raw audio than crash the
    request) if VAD can't be loaded.
    """
    try:
        model, utils = _load_silero()
        get_speech_timestamps = utils[0]
        wav = signal.squeeze(0)
        timestamps = get_speech_timestamps(wav, model, sampling_rate=sample_rate)
        if not timestamps:
            return signal  # let the RMS-floor check downstream reject it
        chunks = [wav[t["start"]:t["end"]] for t in timestamps]
        trimmed = torch.cat(chunks).unsqueeze(0)
        return trimmed
    except Exception as e:  # noqa: BLE001 - deliberately broad, fail-open
        print(f"VAD unavailable, scoring raw audio ({e})")
        return signal
