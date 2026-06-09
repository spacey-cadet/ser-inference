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


# ── Load model at startup ──────────────────────────────────────────────
def load_model():
    from huggingface_hub import snapshot_download
    from speechbrain.integrations.huggingface.wavlm import WavLM
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
        # 1. Load data via soundfile array 
        raw_numpy_array, sample_rate = sf.read(path)
        
        # 2. Map directly to float tensor
        signal = torch.tensor(raw_numpy_array, dtype=torch.float32)
        
        # 3. Structure axis to [channels, time]
        if signal.ndim == 1:
            signal = signal.unsqueeze(0)
        else:
            signal = signal.T

        # 4. Mix down if channels dimension indicates stereo
        # ALTERNATIVE METHOD: Uses .size(0) directly to flush sticky caching
        if signal.size(0) > 1:
            signal = signal.mean(dim=0, keepdim=True)

        # 5. Process resampling sequence
        if sample_rate != 16000:
            signal = process_and_resample_tensor(signal, orig_rate=sample_rate, target_rate=16000)

        # 6. Format to batch structure [1, time] for WavLM input
        final_waveform = signal.squeeze(0).unsqueeze(0)
        wav_lens = torch.tensor([1.0])

        with torch.no_grad():
            features = wavlm(final_waveform)
            if isinstance(features, dict):
                features = features["last_hidden_state"]
            pooled = pooling(features, wav_lens)
            logits = classifier(pooled)
            probs  = torch.softmax(logits, dim=-1).squeeze(0)

        return {
            IDX_TO_EMOTION[i]: round(float(probs[i]), 4)
            for i in range(NUM_CLASSES)
        }
        
    finally:
        if os.path.exists(path):
            os.remove(path)