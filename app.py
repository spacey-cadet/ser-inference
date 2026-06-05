import os
import tempfile

import torch
import torchaudio
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
)

# Load .env for local dev; in Docker, HF_TOKEN comes from runtime env vars
load_dotenv()

MODEL_ID = "space-cadet/wavlm-ser"
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN environment variable is not set")

app = FastAPI()

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = AutoModelForAudioClassification.from_pretrained(MODEL_ID, token=HF_TOKEN)
model.eval()


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        path = tmp.name

    waveform, sr = torchaudio.load(path)

    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    waveform = waveform.mean(dim=0)

    inputs = feature_extractor(
        waveform.numpy(),
        sampling_rate=16000,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # squeeze batch dim: (1, num_classes) -> (num_classes,)
    probs = torch.softmax(outputs.logits, dim=-1)[0]

    return {
        model.config.id2label[i]: round(float(probs[i]), 4)
        for i in range(len(probs))
    }