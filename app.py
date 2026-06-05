import tempfile
from fastapi import FastAPI, File, UploadFile
import torch
import torchaudio
from dotenv import load_all, load_dotenv  # <-- Import dotenv

# Load environment variables from the .env file
load_dotenv()

from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
)

MODEL_ID = "space-cadet/wavlm-ser"

app = FastAPI()

# Transformers automatically picks up the HF_TOKEN from the environment now!
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)

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
        waveform = torchaudio.functional.resample(
            waveform,
            sr,
            16000,
        )

    waveform = waveform.mean(dim=0)

    inputs = feature_extractor(
        waveform.numpy(),
        sampling_rate=16000,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)

    return {
        model.config.id2label[i]: float(probs[i])
        for i in range(probs.shape)
    }