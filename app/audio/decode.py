"""
Format-agnostic audio decoding.

Chat audio arrives as webm/opus, ogg, m4a, wav — whatever the browser's
MediaRecorder or the client SDK produces. PyAV handles all of it uniformly,
which is why this stays as-is from the original inference script rather than
being replaced with a WAV-only path.
"""
import torch
import torch.nn.functional as F
import av


def decode_audio_to_tensor(path: str) -> tuple[torch.Tensor, int]:
    """
    Decodes any audio format (webm/opus, m4a, mp4, mp3, wav, ogg, flac, ...)
    to a float32 tensor of shape [channels, time] using PyAV.
    Returns (tensor, sample_rate).
    """
    container = av.open(path)
    stream = container.streams.audio[0]
    sample_rate = stream.codec_context.sample_rate

    resampler = av.audio.resampler.AudioResampler(format="fltp", layout="stereo", rate=sample_rate)

    frames = []
    for frame in container.decode(stream):
        frame = resampler.resample(frame)
        if isinstance(frame, list):
            for f in frame:
                frames.append(torch.from_numpy(f.to_ndarray()))
        else:
            frames.append(torch.from_numpy(frame.to_ndarray()))

    container.close()

    if not frames:
        raise ValueError("No audio frames decoded")

    signal = torch.cat(frames, dim=1)  # [channels, time]
    return signal, sample_rate


def resample_tensor(audio_tensor: torch.Tensor, orig_rate: int, target_rate: int = 16000) -> torch.Tensor:
    """Pure-torch linear-interpolation resample. Defensive: only runs if rates differ."""
    if orig_rate == target_rate:
        return audio_tensor

    old_length = int(audio_tensor.size(1))
    new_length = int(old_length * (target_rate / orig_rate))

    audio_tensor = audio_tensor.unsqueeze(1)  # [channels, 1, time]
    audio_tensor = F.interpolate(audio_tensor, size=new_length, mode="linear", align_corners=False)
    return audio_tensor.squeeze(1)


def to_mono_16k(signal: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """[channels, time] at any rate -> [1, time] float32 @ 16kHz."""
    if signal.size(0) > 1:
        signal = signal.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        signal = resample_tensor(signal, orig_rate=sample_rate, target_rate=16000)
    return signal
