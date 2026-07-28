"""
Input validation gate for chat audio (Track 2.1).

Chat audio is not RAVDESS audio: shorter, noisier, compressed, variable mic
quality. This module is the defensive layer that runs before a clip ever
reaches the model — reject what shouldn't be scored, and compute the small
feature vector that Track 2.2's drift monitor logs for every request.
"""
from dataclasses import dataclass

import torch


class AudioRejected(Exception):
    """Raised when a clip fails validation. Caller should short-circuit to a
    'no signal' response rather than run inference on it."""
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


@dataclass
class AudioFeatures:
    duration_sec: float
    rms_energy: float
    spectral_centroid: float
    pitch_estimate: float

    def to_dict(self) -> dict:
        return {
            "duration_sec": round(self.duration_sec, 4),
            "rms_energy": round(self.rms_energy, 6),
            "spectral_centroid": round(self.spectral_centroid, 2),
            "pitch_estimate": round(self.pitch_estimate, 2),
        }


def check_duration(signal: torch.Tensor, sample_rate: int, min_sec: float, max_sec: float) -> float:
    duration = signal.size(-1) / sample_rate
    if duration < min_sec:
        raise AudioRejected(f"duration {duration:.2f}s below minimum {min_sec}s")
    if duration > max_sec:
        raise AudioRejected(f"duration {duration:.2f}s exceeds maximum {max_sec}s")
    return duration


def check_not_silent(signal: torch.Tensor, rms_floor: float) -> float:
    rms = torch.sqrt(torch.mean(signal ** 2)).item()
    if rms < rms_floor:
        raise AudioRejected(f"RMS energy {rms:.6f} below silence floor {rms_floor}")
    return rms


def spectral_centroid(signal: torch.Tensor, sample_rate: int) -> float:
    """Cheap, dependency-free spectral centroid via FFT magnitude — enough for
    drift monitoring, not meant to replace librosa for anything precise."""
    x = signal.squeeze(0)
    spectrum = torch.fft.rfft(x)
    magnitudes = spectrum.abs()
    freqs = torch.fft.rfftfreq(x.numel(), d=1.0 / sample_rate)
    total = magnitudes.sum()
    if total <= 0:
        return 0.0
    centroid = (freqs * magnitudes).sum() / total
    return float(centroid.item())


def rough_pitch_estimate(signal: torch.Tensor, sample_rate: int) -> float:
    """Autocorrelation-based pitch estimate — a coarse proxy for drift
    monitoring only. Use librosa.pyin / crepe if you need this for anything
    that isn't a rolling feature-distribution comparison."""
    x = signal.squeeze(0)
    x = x - x.mean()
    if x.abs().sum() == 0:
        return 0.0
    autocorr = torch.nn.functional.conv1d(
        x.view(1, 1, -1), x.view(1, 1, -1).flip(-1), padding=x.numel() - 1
    ).squeeze()
    mid = autocorr.numel() // 2
    min_lag = int(sample_rate / 400)   # ~400 Hz upper bound
    max_lag = int(sample_rate / 60)    # ~60 Hz lower bound
    max_lag = min(max_lag, mid - 1)
    if max_lag <= min_lag:
        return 0.0
    segment = autocorr[mid + min_lag: mid + max_lag]
    if segment.numel() == 0:
        return 0.0
    peak_lag = int(segment.argmax().item()) + min_lag
    if peak_lag == 0:
        return 0.0
    return float(sample_rate / peak_lag)


def validate_and_extract_features(
    signal: torch.Tensor,
    sample_rate: int,
    min_duration_sec: float,
    max_duration_sec: float,
    rms_silence_floor: float,
) -> AudioFeatures:
    """Runs the full validation gate. Raises AudioRejected on failure,
    otherwise returns the feature vector to log for drift monitoring."""
    duration = check_duration(signal, sample_rate, min_duration_sec, max_duration_sec)
    rms = check_not_silent(signal, rms_silence_floor)
    centroid = spectral_centroid(signal, sample_rate)
    pitch = rough_pitch_estimate(signal, sample_rate)
    return AudioFeatures(
        duration_sec=duration,
        rms_energy=rms,
        spectral_centroid=centroid,
        pitch_estimate=pitch,
    )
