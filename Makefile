.PHONY: install dev test drift-check latency benchmark

install:
	pip install -r requirements.txt -r requirements-dev.txt

dev:
	uvicorn app.main:app --reload --port 8000

test:
	pytest tests/ -v

drift-check:
	python scripts/drift_check.py

latency:
	python scripts/benchmark_latency.py --audio-dir data/sample_clips --n 50

calibrate:
	python scripts/calibration_fit.py --predictions data/val_predictions.csv --method isotonic
evaluate:
	    python scripts/eval_report.py --predictions data/val_predictions.csv --out data/eval_report.json