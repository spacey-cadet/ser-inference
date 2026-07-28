# Model Card — WavLM SER `<checkpoint tag/revision>`

Fill this out for every checkpoint pushed to the Hub, champion or
challenger. Push it as the repo's `README.md` on that revision, or as a
`MODEL_CARD.md` alongside `CKPT+*`.

## Summary

- **Base model:** `microsoft/wavlm-base-plus`
- **Classifier head:** `MLPClassifier` (1536 → 256 → 8), see `app/model/classifier.py`
- **Revision / tag:** `<e.g. v3-chat-retrain-2026-08>`
- **Date:** `<YYYY-MM-DD>`
- **Trained by:** `<name/team>`

## Training data

- **Source corpora:** RAVDESS / IEMOCAP / CREMA-D / `<chat-domain holdout, once it exists>`
- **Retrain cycle:** `<e.g. "retrain #2">`
- **Chat-domain examples included:** `<count>` — pulled from the review
  queue (`/admin/review-queue/export`), specifically the examples the prior
  checkpoint scored low-confidence on.
- **Class distribution:** `<paste label_counts from scripts/ge_validate_dataset.py>`

## Hyperparameters

- Learning rate: `<...>`
- Epochs: `<...>`
- Batch size: `<...>`
- Optimizer: `<...>`
- Frozen layers: `<...>`

## Eval metrics (lead with per-class, not aggregate accuracy)

| Corpus | Accuracy | Notes |
|---|---|---|
| RAVDESS (in-distribution) | `<...>` | |
| IEMOCAP (cross-corpus) | `<...>` | |
| CREMA-D (cross-corpus) | `<...>` | |
| Chat-domain holdout | `<...>` | Most important once it exists — reflects true deployment conditions |

Per-class F1 / PR-AUC (from `scripts/eval_report.py`):

| Emotion | F1 | PR-AUC | Support |
|---|---|---|---|
| angry | | | |
| calm | | | |
| disgust | | | |
| fearful | | | |
| happy | | | |
| neutral | | | |
| sad | | | |
| surprised | | | |

Flag the weakest class(es) explicitly here — don't bury them in the table.

## Calibration

- Method: `<platt / isotonic / none>`
- Reliability diagram: `<link to data/calibration/reliability_pre.png and reliability_post.png>`
- Thresholds in use: `THRESHOLD_LOW=<...>`, `THRESHOLD_HIGH=<...>`

## Latency (from `scripts/benchmark_latency.py`)

- Hardware: `<free-tier HF Spaces CPU, e.g. 2 vCPU>`
- p50: `<...> ms`
- p95: `<...> ms`
- p99: `<...> ms`
- Batch size: 1 (single request, no batching)

## Champion/challenger comparison (if this is a challenger)

- Compared against: `<champion revision>`
- Result: `<PROMOTE / DO NOT PROMOTE>` (from `scripts/champion_challenger_eval.py`)
- Corpora it cleared the bar on: `<list>`
- Corpora it did NOT clear the bar on (if any): `<list — be honest here>`

## Known limitations

`<e.g. "Sad/Neutral confusion persists in this revision; see per-class F1
above." / "No chat-domain holdout yet — cross-corpus numbers are the best
proxy for chat-audio performance until the review queue produces enough
labeled examples.">`
