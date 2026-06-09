import os
import tempfile
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import av
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile

load_dotenv()

MODEL_ID   = "space-cadet/wavlm-ser"
HF_TOKEN   = os.environ.get("HF_TOKEN")
CACHE_DIR  = "./model_cache"

IDX_TO_EMOTION = {
    0: "angry", 1: "calm", 2: "disgust", 3: "fearful",
    4: "happy", 5: "neutral", 6: "sad", 7: "surprised",
}
NUM_CLASSES = 8

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN environment variable is not set")


# ── Model definition (must match training) ─────────────────────────────
class MLPClassifier(nn.Module):
    def __init__(self, input_size=1536, hidden=256, n_classes=8, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        return self.net(x)


# ── Resample Core ──────────────────────────────────────────────────────
def process_and_resample_tensor(audio_tensor, orig_rate, target_rate=16000):
    """
    Handles interpolation resampling safely using pure torch operations.
    """
    if orig_rate == target_rate:
        return audio_tensor

    old_length = int(audio_tensor.size(1))
    new_length = int(old_length * (target_rate / orig_rate))

    # Format to 3D for interpolation: [channels, 1, time]
    audio_tensor = audio_tensor.unsqueeze(1)
    audio_tensor = F.interpolate(audio_tensor, size=new_length, mode='linear', align_corners=False)
    return audio_tensor.squeeze(1)


# ── Audio Decoder ──────────────────────────────────────────────────────
def decode_audio_to_tensor(path: str) -> tuple[torch.Tensor, int]:
    """
    Decodes any audio format (m4a, mp4, mp3, wav, flac, etc.)
    to a float32 tensor of shape [channels, time] using PyAV.
    Returns (tensor, sample_rate).
    """
    container = av.open(path)
    stream = container.streams.audio[0]
    sample_rate = stream.codec_context.sample_rate

    resampler = av.audio.resampler.AudioResampler(format='fltp', layout='stereo', rate=sample_rate)

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


# ── Load model at startup ──────────────────────────────────────────────
def load_model():
    from huggingface_hub import snapshot_download
    from speechbrain.lobes.models.huggingface_transformers.wavlm import WavLM
    from speechbrain.nnet.pooling import StatisticsPooling

    local_dir = snapshot_download(
        repo_id=MODEL_ID,
        repo_type="model",
        token=HF_TOKEN,
        local_dir=CACHE_DIR,
        ignore_patterns=["pretrained_models/**"],
    )

    ckpt_dirs = sorted(Path(local_dir).glob("CKPT+*"))
    if not ckpt_dirs:
        raise FileNotFoundError(f"No CKPT+* folder found in {local_dir}")
    ckpt_dir = ckpt_dirs[-1]
    print(f"Loading checkpoint: {ckpt_dir.name}")

    wavlm = WavLM(
        source="microsoft/wavlm-base-plus",
        save_path=os.path.join(CACHE_DIR, "pretrained_models/wavlm"),
        output_norm=True,
        freeze=True,
        freeze_feature_extractor=True,
    )

    wavlm_ckpt = ckpt_dir / "wavlm.ckpt"
    if wavlm_ckpt.exists():
        state = torch.load(str(wavlm_ckpt), map_location="cpu", weights_only=True)
        wavlm.load_state_dict(state, strict=False)
        print("WavLM weights loaded")

    pooling    = StatisticsPooling()
    classifier = MLPClassifier()

    clf_ckpt = ckpt_dir / "classifier.ckpt"
    if clf_ckpt.exists():
        state = torch.load(str(clf_ckpt), map_location="cpu", weights_only=True)
        print(f"Classifier keys in checkpoint: {list(state.keys())}")
        classifier.load_state_dict(state, strict=True)
        print("Classifier weights loaded successfully")
    else:
        raise FileNotFoundError(f"classifier.ckpt not found in {ckpt_dir}")

    wavlm.eval()
    pooling.eval()
    classifier.eval()

    return wavlm, pooling, classifier


def verify_model(wavlm, pooling, classifier):
    """Sanity check — a trained model should not produce uniform ~0.125 probabilities."""
    dummy = torch.randn(1, 16000)
    with torch.no_grad():
        feat = wavlm(dummy)
        if isinstance(feat, dict):
            feat = feat["last_hidden_state"]
        pooled = pooling(feat, torch.tensor([1.0]))
        logits = classifier(pooled)
        probs  = torch.softmax(logits, dim=-1).squeeze(0)

    max_conf = probs.max().item()
    pred     = probs.argmax().item()
    print(f"Sanity check → predicted: {IDX_TO_EMOTION[pred]}, confidence: {max_conf:.3f}")

    if max_conf < 0.2:
        raise RuntimeError(
            f"Model looks uninitialised (max_conf={max_conf:.3f}). "
            "Check that classifier.ckpt keys match MLPClassifier."
        )


wavlm, pooling, classifier = load_model()
verify_model(wavlm, pooling, classifier)

app = FastAPI()


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix if file.filename else ".audio"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        path = tmp.name

    try:
        # 1. Decode any format → [channels, time] float32 tensor
        signal, sample_rate = decode_audio_to_tensor(path)

        # 2. Mix down stereo → mono
        if signal.size(0) > 1:
            signal = signal.mean(dim=0, keepdim=True)

        # 3. Resample to 16kHz if needed
        if sample_rate != 16000:
            signal = process_and_resample_tensor(signal, orig_rate=sample_rate, target_rate=16000)

        # 4. Shape to [1, time] for WavLM
        final_waveform = signal.squeeze(0).unsqueeze(0)
        wav_lens = torch.tensor([1.0])

        with torch.no_grad():
            features = wavlm(final_waveform)
            if isinstance(features, dict):
                features = features["last_hidden_state"]
            pooled = pooling(features, wav_lens)
            logits = classifier(pooled)
            probs  = torch.softmax(logits, dim=-1).squeeze(0)

        probs_list = probs.tolist()
        ranked = sorted(
            [{"emotion": IDX_TO_EMOTION[i], "score": round(probs_list[i], 4)} for i in range(NUM_CLASSES)],
            key=lambda x: x["score"],
            reverse=True,
        )

        return {
            "top_emotion": ranked[0]["emotion"],
            "ranking": ranked,
        }

    finally:
        if os.path.exists(path):
            os.remove(path)