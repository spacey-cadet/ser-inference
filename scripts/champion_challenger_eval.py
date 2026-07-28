"""
Track 1.5 — champion/challenger comparison, simulated offline.

Loads both the current (champion) and a retrained (challenger) checkpoint,
runs both over the same held-out and cross-corpus sets, and only recommends
promotion if the challenger clears the champion's bar on ALL corpora — not
just RAVDESS. Once real chat-audio samples accumulate through the Track 2
label queue, add that set to the comparison; it becomes the most important
one since it's the only one reflecting true deployment conditions.

Usage:
    python scripts/champion_challenger_eval.py \
        --champion-revision main --challenger-revision v2 \
        --corpora ravdess=data/ravdess_val.csv iemocap=data/iemocap_val.csv \
                  cremad=data/cremad_val.csv chat_holdout=data/chat_holdout.csv

Each corpus CSV needs: path (to audio), true_label (0-7 index).
This script calls app.model.loader.load_model directly (not the FastAPI
app) since it's an offline batch job, not a serving path.
"""
import argparse
import json

import pandas as pd
import torch

from app.audio.decode import decode_audio_to_tensor, to_mono_16k
from app.model.loader import load_model, verify_model
from app.model.classifier import NUM_CLASSES
from app.config import get_settings


def eval_corpus(model, csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    correct = 0
    for _, row in df.iterrows():
        signal, sr = decode_audio_to_tensor(row["path"])
        signal = to_mono_16k(signal, sr)
        waveform = signal.squeeze(0).unsqueeze(0)
        with torch.no_grad():
            feat = model.wavlm(waveform)
            if isinstance(feat, dict):
                feat = feat["last_hidden_state"]
            pooled = model.pooling(feat, torch.tensor([1.0]))
            logits = model.classifier(pooled)
            pred = logits.argmax(dim=-1).item()
        correct += int(pred == int(row["true_label"]))
    return {"n": len(df), "accuracy": correct / len(df) if len(df) else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--champion-revision", default="main")
    ap.add_argument("--challenger-revision", required=True)
    ap.add_argument("--corpora", nargs="+", required=True, help="name=path.csv pairs")
    ap.add_argument("--out", default="data/champion_challenger_report.json")
    args = ap.parse_args()

    settings = get_settings()
    corpora = dict(pair.split("=", 1) for pair in args.corpora)

    champion = load_model(settings.model_id, settings.hf_token, settings.cache_dir, args.champion_revision)
    verify_model(champion)
    challenger = load_model(settings.model_id, settings.hf_token, settings.cache_dir, args.challenger_revision)
    verify_model(challenger)

    report = {"champion": {}, "challenger": {}}
    for name, path in corpora.items():
        print(f"Evaluating champion on {name}...")
        report["champion"][name] = eval_corpus(champion, path)
        print(f"Evaluating challenger on {name}...")
        report["challenger"][name] = eval_corpus(challenger, path)

    clears_bar = all(
        report["challenger"][c]["accuracy"] >= report["champion"][c]["accuracy"]
        for c in corpora
        if report["champion"][c]["accuracy"] is not None
    )
    report["recommendation"] = "PROMOTE" if clears_bar else "DO NOT PROMOTE"
    report["note"] = "Promotion requires clearing the bar on every corpus, not just the easiest one."

    print(json.dumps(report, indent=2))
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
