"""
kaggle/train_head_kernel.py

Runs ON KAGGLE (GPU, no internet — see kernel-metadata.json). Consumes
three pinned dataset versions attached to the kernel:
  - ser-retrain-batch:      this run's new labeled audio + manifest.csv
                             (filename, label, request_id — label is a
                             canonical lowercase emotion string)
  - ser-eval-set-pinned:    the FIXED held-out set — never changes
  - ser-embeddings-cache:   accumulated WavLM embeddings from prior runs

CANONICAL LABEL SCHEME — must match scripts/eval_report.py's
IDX_TO_EMOTION EXACTLY. Duplicated here (not imported) because this
runs in the Kaggle sandbox, a different environment than the
GH-Actions-run eval_report.py — if you later attach the repo itself as
a kernel dataset_source and import app.emotion_labels or similar from
both places instead, update this comment and remove the duplication.
If you change one of these dicts, you MUST change the other.
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

sys.path.insert(0, "/kaggle/input/ser-inference-src")  # only used for eval_report.py subprocess call below

EMOTION_TO_IDX = {
    "angry": 0, "calm": 1, "disgust": 2, "fearful": 3,
    "happy": 4, "neutral": 5, "sad": 6, "surprised": 7,
}
IDX_TO_EMOTION = {v: k for k, v in EMOTION_TO_IDX.items()}
NUM_CLASSES = 8

BATCH_DIR = Path("/kaggle/input/ser-retrain-batch")
EVAL_DIR = Path("/kaggle/input/ser-eval-set-pinned")
CACHE_DIR = Path("/kaggle/input/ser-embeddings-cache")
OUT_DIR = Path("/kaggle/working")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_frozen_wavlm():
    from transformers import WavLMModel
    model = WavLMModel.from_pretrained(
        "microsoft/wavlm-large", revision="<pin-exact-revision-here>"
    )
    model.eval().to(DEVICE)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def encode_label(label_str: str) -> int:
    """Raises loudly on an unrecognized label string, rather than
    silently corrupting the training set — a typo in the admin UI's
    submitted label (or a future label taxonomy change) should fail
    the kernel run, not train on garbage."""
    key = label_str.strip().lower()
    if key not in EMOTION_TO_IDX:
        raise ValueError(
            f"Unrecognized label '{label_str}' — must be one of "
            f"{sorted(EMOTION_TO_IDX)}. Check the admin UI's submitted "
            f"value and scripts/eval_report.py's IDX_TO_EMOTION for drift."
        )
    return EMOTION_TO_IDX[key]


def extract_embeddings(wavlm, manifest_rows, audio_dir):
    embeddings, label_indices, request_ids = [], [], []
    with torch.no_grad():
        for row in manifest_rows:
            wav, sr = torchaudio.load(audio_dir / row["filename"])
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            wav = wav.to(DEVICE)
            out = wavlm(wav).last_hidden_state.mean(dim=1)
            embeddings.append(out.squeeze(0).cpu().numpy())
            label_indices.append(encode_label(row["label"]))
            request_ids.append(row["request_id"])
    return np.stack(embeddings), np.array(label_indices), request_ids


def expand_proba_to_full_width(proba: np.ndarray, classes_seen: np.ndarray) -> np.ndarray:
    """sklearn's predict_proba only returns columns for classes present
    at fit time. If the training batch didn't contain all 8 emotions,
    proba.shape[1] < 8 and the columns won't align with prob_0..prob_7.
    This pads with zero-probability columns for any class absent from
    training, so the output CSV always has exactly 8 prob columns in
    the canonical index order eval_report.py expects."""
    full = np.zeros((proba.shape[0], NUM_CLASSES), dtype=proba.dtype)
    for col_idx, class_label in enumerate(classes_seen):
        full[:, int(class_label)] = proba[:, col_idx]
    return full


def main():
    wavlm = load_frozen_wavlm()

    manifest_rows = list(csv.DictReader(open(BATCH_DIR / "manifest.csv")))
    new_embeddings, new_label_idx, new_ids = extract_embeddings(
        wavlm, manifest_rows, BATCH_DIR / "audio"
    )

    cache_path = CACHE_DIR / "embeddings_cache.npz"
    if cache_path.exists():
        cache = np.load(cache_path, allow_pickle=True)
        all_embeddings = np.concatenate([cache["embeddings"], new_embeddings])
        all_label_idx = np.concatenate([cache["label_idx"], new_label_idx])
    else:
        all_embeddings, all_label_idx = new_embeddings, new_label_idx

    np.savez(
        OUT_DIR / "updated_embeddings_cache.npz",
        embeddings=all_embeddings,
        label_idx=all_label_idx,
    )

    # --- retrain the MLP head ---
    # ADJUST to match app/model/classifier.py's actual MLPClassifier
    # definition if it differs from this sklearn-style stand-in.
    from sklearn.neural_network import MLPClassifier

    head = MLPClassifier(hidden_layer_sizes=(128,), max_iter=500)
    head.fit(all_embeddings, all_label_idx)

    import joblib
    joblib.dump({"model": head, "classes_seen": head.classes_}, OUT_DIR / "challenger_head.pt")

    # --- evaluate against the PINNED held-out set ---
    eval_manifest = list(csv.DictReader(open(EVAL_DIR / "manifest.csv")))
    eval_embeddings, eval_label_idx, eval_ids = extract_embeddings(
        wavlm, eval_manifest, EVAL_DIR / "audio"
    )

    raw_proba = head.predict_proba(eval_embeddings)
    full_proba = expand_proba_to_full_width(raw_proba, head.classes_)

    predictions_csv = OUT_DIR / "val_predictions.csv"
    with open(predictions_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["request_id", "true_label"] + [f"prob_{i}" for i in range(NUM_CLASSES)])
        for rid, true_idx, proba_row in zip(eval_ids, eval_label_idx, full_proba):
            writer.writerow([rid, int(true_idx)] + [f"{p:.6f}" for p in proba_row])

    # Reuse eval_report.py UNMODIFIED — repo must be attached as a
    # kernel dataset_source at /kaggle/input/ser-inference-src for this
    # subprocess call to find it.
    subprocess.run(
        [
            "python", "/kaggle/input/ser-inference-src/scripts/eval_report.py",
            "--predictions", str(predictions_csv),
            "--out", str(OUT_DIR / "challenger_eval_report.json"),
            "--corpus", "kaggle-retrain-challenger",
        ],
        check=True,
    )

    print("Done. Outputs in /kaggle/working/:")
    for f in OUT_DIR.iterdir():
        print(" -", f.name)


if __name__ == "__main__":
    main()