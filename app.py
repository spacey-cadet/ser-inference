import os
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import soundfile as sf
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
    def __init__(self, input_size=1536, n_classes=8):
        super().__init__()
        self.w = nn.Linear(input_size, n_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        return self.w(x)

# ── Resampling Helper Function ─────────────────────────────────────────
def resample_audio(wav, orig_freq, new_freq):
    """Resamples a 2D torch tensor [channels, time] using linear interpolation."""
    if orig_freq == new_freq:
        return wav
    
    # Calculate new length
    orig_len = wav.shape
    new_len = int(orig_len * (new_freq / orig_freq))
    
    # unsqueeze to [channels, 1, time] for interpolate
    wav = wav.unsqueeze(1)
    wav = F.interpolate(wav, size=new_len, mode='linear', align_corners=False)
    return wav.squeeze(1)

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

    pooling    = StatisticsPooling()
    classifier = MLPClassifier()

    clf_ckpt = ckpt_dir / "classifier.ckpt"
    if clf_ckpt.exists():
        state = torch.load(str(clf_ckpt), map_location="cpu", weights_only=True)
        classifier.load_state_dict(state)

    wavlm.eval()
    pooling.eval()
    classifier.eval()
    return wavlm, pooling, classifier


wavlm, pooling, classifier = load_model()

app = FastAPI()


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        path = tmp.name

    try:
        # Load audio via soundfile
        wav_np, sr = sf.read(path)
        
        # Convert to torch tensor
        wav = torch.tensor(wav_np, dtype=torch.float32)
        
        # Ensure shape is [channels, time]
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)  # [1, T] for mono
        else:
            wav = wav.T             # Transpose to [channels, T] if stereo

        # Handle stereo downmixing
        if wav.shape > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Handle resampling
        if sr != 16000:
            wav = resample_audio(wav, orig_freq=sr, new_freq=16000)

        # Standardize shape to [1, T] for model processing
        wav = wav.squeeze(0).unsqueeze(0)
        wav_lens = torch.tensor([1.0])

        with torch.no_grad():
            features = wavlm(wav)
            if isinstance(features, dict):
                features = features["last_hidden_state"]
            pooled = pooling(features, wav_lens)
            logits = classifier(pooled)
            probs  = torch.softmax(logits, dim=-1).squeeze(0)  #

        return {
            IDX_TO_EMOTION[i]: round(float(probs[i]), 4)
            for i in range(NUM_CLASSES)
        }
        
    finally:
        # Ensure temporary file is cleaned up
        if os.path.exists(path):
            os.remove(path)