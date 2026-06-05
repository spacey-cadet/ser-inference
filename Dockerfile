FROM python:3.11

WORKDIR /app

COPY requirements.txt .

# speechbrain needs numpy installed first to avoid build issues
RUN pip install --no-cache-dir numpy
RUN pip install --no-cache-dir speechbrain
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD uvicorn app:app --host 0.0.0.0 --port 7860