"""
Track 3.4 — explicit p50/p99 inference latency benchmark.

This number should appear next to accuracy numbers in the model card going
forward. Free-tier HF Spaces are CPU-only and a WavLM-base forward pass is
not free there — measure it, don't assume it's negligible.

Usage:
    python scripts/benchmark_latency.py --audio-dir data/sample_clips --n 50
"""
import argparse
import glob
import time

import numpy as np
import torch

from app.audio.decode import decode_audio_to_tensor, to_mono_16k
from app.calibration.calibrate import CalibrationMap
from app.config import get_settings
from app.inference.predict import predict_from_path
from app.model.registry import ModelRegistry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", required=True, help="Directory of sample audio clips to benchmark against")
    ap.add_argument("--n", type=int, default=50, help="Number of inference calls to time")
    args = ap.parse_args()

    settings = get_settings()
    registry = ModelRegistry.bootstrap(settings)
    calibration = CalibrationMap.load(settings.calibration_path)

    files = glob.glob(f"{args.audio_dir}/*")
    if not files:
        raise SystemExit(f"No audio files found in {args.audio_dir}")

    latencies = []
    for i in range(args.n):
        path = files[i % len(files)]
        outcome = predict_from_path(path, registry.champion, calibration, settings)
        latencies.append(outcome.latency_ms)
        print(f"[{i+1}/{args.n}] {outcome.latency_ms:.1f} ms")

    latencies = np.array(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)

    print("\n--- Latency summary (CPU, single-request, no batching) ---")
    print(f"p50: {p50:.1f} ms")
    print(f"p95: {p95:.1f} ms")
    print(f"p99: {p99:.1f} ms")
    print(f"mean: {latencies.mean():.1f} ms  |  max: {latencies.max():.1f} ms")
    print("\nRecord these in the model card (docs/MODEL_CARD_TEMPLATE.md) alongside accuracy metrics.")
    print("If this is on the critical path of the chat response, consider ONNX export, "
          "a smaller/quantized checkpoint, or decoupling scoring from the response path.")


if __name__ == "__main__":
    main()
