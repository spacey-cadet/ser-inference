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

# Load environment variables from the .env file
load_dotenv()

MODEL_ID = "space-cadet/wavlm-ser"
HF_TOKEN = os.getenv("HF_TOKEN")

app = FastAPI()

# Pass token explicitly so it works across all transformers versions
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = AutoModelForAudioClassification.from_pretrained(MODEL_ID, token=HF_TOKEN)
model.eval()


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        path = tmp.name

    waveform, sr = torchaudio.load(path)

    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    # Mix down to mono
    waveform = waveform.mean(dim=0)

    inputs = feature_extractor(
        waveform.numpy(),
        sampling_rate=16000,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # Remove batch dim before indexing: shape is (1, num_classes) → (num_classes,)
    probs = torch.softmax(outputs.logits, dim=-1)[0]

    return {
        model.config.id2label[i]: float(probs[i])
        for i in range(len(probs))
    }