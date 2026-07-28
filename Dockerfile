FROM python:3.11

WORKDIR /app

COPY requirements.txt .

# speechbrain needs numpy installed first to avoid build issues
RUN pip install --no-cache-dir numpy
RUN pip install --no-cache-dir speechbrain
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Persistent-storage-friendly default paths; override via env vars if the
# Space has a mounted persistent volume elsewhere.
ENV CACHE_DIR=/app/model_cache
ENV SQLITE_PATH=/app/data/telemetry.db
ENV CALIBRATION_PATH=/app/data/calibration/calibration.json
ENV DRIFT_REFERENCE_PATH=/app/data/drift_reference/train_features.json

CMD uvicorn app.main:app --host 0.0.0.0 --port 7860
