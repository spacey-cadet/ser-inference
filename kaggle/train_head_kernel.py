"""
kaggle/train_head_kernel.py

Runs ON KAGGLE (GPU, no internet — see kernel-metadata.json). Consumes
three pinned dataset versions attached to the kernel:
  - ser-retrain-batch:      this run's new labeled audio + manifest.csv
                             (from scripts/assemble_kaggle_batch.py)
  - ser-eval-set-pinned:    the FIXED held-out set — never changes
                             between runs, or champion/challenger scores
                             aren't comparable
  - ser-embeddings-cache:   accumulated WavLM embeddings from all prior
                             training data, so this doesn't recompute
                             embeddings for old data every run

Only the MLP head is trained here — WavLM is loaded frozen, eval()
mode, no_grad for the forward pass. That's what makes this a CPU-cheap,
GPU-optional job in principle; it's on Kaggle GPU mainly because the
embedding extraction forward pass over the full accumulated dataset is
faster there than on a GH Actions CPU runner, not because backprop
through WavLM is happening.

Outputs (left in /kaggle/working/, which `kaggle kernels output` pulls
back to GH Actions):
  - challenger_head.pt              new MLP head weights
  - challenger_eval_report.json     eval_report.py's output, scored
                                     against ser-eval-set-pinned
  - updated_embeddings_cache.npz    merged cache for next run's reuse

NOTE: I have not seen your actual classifier.py / eval_report.py
source, so the WavLM loading, MLPClassifier construction, and
eval_report.py invocation below are written against the interfaces
implied by PIPELINE_README.md's descriptions. Check these against your
real `app/model/classifier.py` and `scripts/eval_report.py` before
trusting kernel output.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

sys.path.insert(0, "/kaggle/input/ser-inference-src")  # if you attach repo code as a dataset too

BATCH_DIR = Path("/kaggle/input/ser-retrain-batch")
EVAL_DIR = Path("/kaggle/input/ser-eval-set-pinned")
CACHE_DIR = Path("/kaggle/input/ser-embeddings-cache")
OUT_DIR = Path("/kaggle/working")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_frozen_wavlm():
    from transformers import WavLMModel
    # Pin the exact revision that matches the currently-deployed head —
    # a mismatched backbone silently produces a different embedding
    # space and the head's weights become meaningless.
    model = WavLMModel.from_pretrained(
        "microsoft/wavlm-large", revision="<pin-exact-revision-here>"
    )
    model.eval().to(DEVICE)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def extract_embeddings(wavlm, manifest_rows, audio_dir):
    embeddings, labels, request_ids = [], [], []
    with torch.no_grad():
        for row in manifest_rows:
            wav, sr = torchaudio.load(audio_dir / row["filename"])
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            wav = wav.to(DEVICE)
            out = wavlm(wav).last_hidden_state.mean(dim=1)  # mean-pool over time
            embeddings.append(out.squeeze(0).cpu().numpy())
            labels.append(row["label"])
            request_ids.append(row["request_id"])
    return np.stack(embeddings), labels, request_ids


def main():
    import csv

    wavlm = load_frozen_wavlm()

    # --- new batch ---
    manifest_rows = list(csv.DictReader(open(BATCH_DIR / "manifest.csv")))
    new_embeddings, new_labels, new_ids = extract_embeddings(
        wavlm, manifest_rows, BATCH_DIR / "audio"
    )

    # --- merge with cached embeddings from prior runs ---
    cache_path = CACHE_DIR / "embeddings_cache.npz"
    if cache_path.exists():
        cache = np.load(cache_path, allow_pickle=True)
        all_embeddings = np.concatenate([cache["embeddings"], new_embeddings])
        all_labels = list(cache["labels"]) + new_labels
    else:
        all_embeddings, all_labels = new_embeddings, new_labels

    np.savez(
        OUT_DIR / "updated_embeddings_cache.npz",
        embeddings=all_embeddings,
        labels=np.array(all_labels),
    )

    # --- retrain the MLP head ---
    # ADJUST to match app/model/classifier.py's actual MLPClassifier
    # definition — this assumes a simple sklearn-style class for
    # brevity; your real head may be a torch.nn.Module trained with an
    # explicit loop instead.
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y = le.fit_transform(all_labels)
    head = MLPClassifier(hidden_layer_sizes=(128,), max_iter=500)
    head.fit(all_embeddings, y)

    import joblib
    joblib.dump({"model": head, "label_encoder": le}, OUT_DIR / "challenger_head.pt")

    # --- evaluate against the PINNED held-out set ---
    eval_manifest = list(csv.DictReader(open(EVAL_DIR / "manifest.csv")))
    eval_embeddings, eval_labels, eval_ids = extract_embeddings(
        wavlm, eval_manifest, EVAL_DIR / "audio"
    )
    eval_pred_idx = head.predict(eval_embeddings)
    eval_pred_labels = le.inverse_transform(eval_pred_idx)

    predictions_csv = OUT_DIR / "val_predictions.csv"
    with open(predictions_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["request_id", "true_label", "predicted_label"])
        for rid, true_l, pred_l in zip(eval_ids, eval_labels, eval_pred_labels):
            writer.writerow([rid, true_l, pred_l])

    # Reuse eval_report.py UNMODIFIED so champion and challenger are
    # scored with identical logic — this is the whole point of keeping
    # it a separate reporting step.
    subprocess.run(
        [
            "python", "/kaggle/input/ser-inference-src/scripts/eval_report.py",
            "--predictions", str(predictions_csv),
            "--out", str(OUT_DIR / "challenger_eval_report.json"),
        ],
        check=True,
    )

    print("Done. Outputs in /kaggle/working/:")
    for f in OUT_DIR.iterdir():
        print(" -", f.name)


if __name__ == "__main__":
    main()
