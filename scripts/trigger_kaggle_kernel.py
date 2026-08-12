#!/usr/bin/env python
"""
scripts/trigger_kaggle_kernel.py

Pushes and runs the training kernel (kaggle/train_head_kernel.py) via
the Kaggle API, polls until it finishes, and pulls the output files
(challenger weights + challenger_eval_report.json) back into the GH
Actions workspace for the promotion gate to consume.

Requires KAGGLE_USERNAME / KAGGLE_KEY in the environment (GitHub
Secrets), and a kernel-metadata.json already present alongside the
kernel script (kaggle/kernel-metadata.json) declaring the pinned
training-batch dataset + pinned eval-set dataset as dataset_sources.

Usage:
    python scripts/trigger_kaggle_kernel.py \
        --kernel-dir kaggle/ \
        --poll-interval 30 --timeout 1800 \
        --out-dir /tmp/kaggle-output
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"command failed: {' '.join(cmd)}")
    return result.stdout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kernel-dir", required=True)
    p.add_argument("--poll-interval", type=int, default=30)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    print("Pushing kernel to Kaggle...")
    run(["kaggle", "kernels", "push", "-p", args.kernel_dir])

    meta = json.loads((Path(args.kernel_dir) / "kernel-metadata.json").read_text())
    kernel_slug = meta["id"]  # e.g. "your-username/ser-retrain-head"

    print(f"Polling {kernel_slug} for completion...")
    elapsed = 0
    while elapsed < args.timeout:
        status_out = run(["kaggle", "kernels", "status", kernel_slug])
        print(status_out.strip())
        if "complete" in status_out.lower():
            break
        if "error" in status_out.lower() or "failed" in status_out.lower():
            raise RuntimeError(f"Kaggle kernel run failed: {status_out}")
        time.sleep(args.poll_interval)
        elapsed += args.poll_interval
    else:
        raise TimeoutError(f"Kernel {kernel_slug} did not finish within {args.timeout}s")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    print("Pulling kernel output...")
    run(["kaggle", "kernels", "output", kernel_slug, "-p", args.out_dir])
    print(f"Output pulled to {args.out_dir} — expect challenger weights + "
          f"challenger_eval_report.json there.")


if __name__ == "__main__":
    main()
