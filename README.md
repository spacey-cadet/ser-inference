# WavLM SER — Production Pipeline

This repo turns the single-endpoint `ser-inference` service into the
production-grade pipeline described in `docs/PLAN.md`, built entirely on
free-tier infrastructure (Hugging Face Spaces + GitHub Actions + SQLite/HF
Datasets), and mapped directly to Track 1 (modeling rigor/calibration),
Track 2 (data quality/drift/retraining loop), and Track 3
(rollout/registry/session-state/latency) from the pipeline guide.

## CHANGES `app.py`

The original service did one thing: decode audio → WavLM → classifier →
return ranked emotions. It's still in here, functionally intact — but it's
now one path through a pipeline that also:

- calibrates raw softmax scores before thresholding (Track 1.2)
- runs a two-threshold confidence cascade with a chat-safe neutral default
  (Track 1.3)
- validates and VAD-trims audio before scoring (Track 2.1)
- logs per-request features for drift monitoring and a review/label queue
  (Track 2.2 / 2.3), gated on consent (Track 2.4)
- maintains session-level rolling emotion state across chat turns (Track 3.3)
- supports in-process champion/challenger canary routing (Track 3.2)
- reports and gates on latency (Track 3.4)

Everything above is free — no paid infra required. See
`docs/PLAN.md` → "What still needs paid infrastructure" 

## Repository layout

```
app/
  config.py               Central settings (env-var driven)
  main.py                 FastAPI app + startup wiring
  api/routes.py            /predict, /, /admin/review-queue/*
  model/
    classifier.py          MLPClassifier head (must match training)
    loader.py               HF Hub revision-aware model loading + sanity check
    registry.py             Champion/challenger holder
  audio/
    decode.py               PyAV decode + resample (from original app.py)
    validate.py              Duration/RMS gate + drift feature extraction
    vad.py                   Silence trimming (silero-vad)
  calibration/
    calibrate.py              Applies Platt/isotonic mapping fit offline
  inference/
    cascade.py                 Two-threshold confidence cascade
    predict.py                  Orchestrates decode->validate->VAD->model->calibrate->cascade
  session/
    state.py                    Session-keyed rolling emotion state (memory/Redis)
  logging_pipeline/
    feature_log.py               Per-request feature logging (SQLite/HF Dataset)
    hf_dataset_log.py             HF Dataset backend
    review_queue.py                Low-confidence queue helpers
    consent.py                     Consent gate before any logging
  routing/canary.py               In-process champion/challenger traffic split
  alerts/webhook.py                Slack/Discord webhook

scripts/                     Offline jobs (not part of the request path)
  baseline_classical_model.py    Track 1.1 classical baseline
  calibration_fit.py              Track 1.2 fit Platt/isotonic + reliability diagrams
  eval_report.py                   Track 1.4 imbalance-aware eval (PR-AUC, per-class F1)
  champion_challenger_eval.py       Track 1.5 offline champion/challenger comparison
  build_drift_reference.py          Builds the training-distribution reference file
  drift_check.py                     Track 2.2 KS-test drift check + alerting
  ge_validate_dataset.py              Track 2 one-time Great Expectations gate
  benchmark_latency.py                 Track 3.4 p50/p99 latency benchmark

.github/workflows/
  drift-check.yml              Scheduled drift check (GitHub Actions cron)
  ci-eval-gate.yml               Tests + eval gate before promoting to prod

docs/
  PLAN.md                       Phased build roadmap (mirrors the pipeline guide)
  ARCHITECTURE.md                 Request-flow diagram + component responsibilities
  CONSENT_RETENTION_POLICY.md      Track 2.4 policy template — publish before logging goes live
  ROLLOUT.md                        Canary / blue-green / shadow deployment playbook
  MODEL_CARD_TEMPLATE.md             Fill this out per checkpoint pushed to the Hub

tests/                        Unit tests for the pure-logic pieces (cascade, validation, consent, session state)
```

## Quickstart (local dev)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env   # fill in HF_TOKEN at minimum
uvicorn app.main:app --reload --port 8000
```

Test it:

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@testfile.wav" \
  -F "session_id=demo-session-1" \
  -F "consent=true"
```

Run the test suite (no model download required — these test the pure logic):

```bash
pytest tests/ -v
```

## Deploying to Hugging Face Spaces (free tier)

1. Push this repo to a new Space with `sdk: docker`. HF Spaces reads its
   config from `README.md`'s front-matter, which collides with this repo's
   GitHub-facing README — `SPACE_README.md` has that front-matter block
   ready to go; rename it to `README.md` (or copy the front-matter block
   into this one) when pushing to the Space.
2. Set repo secrets / Space variables: `HF_TOKEN` at minimum. Everything
   else in `.env.example` has a working default.
3. First deploy will run uncalibrated (`calibration.method == "none"`) and
   without a challenger — that's expected. Phase 1 in `docs/PLAN.md` covers
   getting calibration live.

## Where each Track-1/2/3 item in the guide lives

| Guide item | Code / doc |
|---|---|$$
| 1.1 Classical baseline | `scripts/baseline_classical_model.py` |
| 1.2 Calibration | `scripts/calibration_fit.py` (offline fit) → `app/calibration/calibrate.py` (applied online) |
| 1.3 Confidence cascade | `app/inference/cascade.py` |
| 1.4 Imbalance-aware eval | `scripts/eval_report.py` |
| 1.5 Champion/challenger | `scripts/champion_challenger_eval.py` (offline) → `app/routing/canary.py` (online) |
| 2.1 Input validation gate | `app/audio/validate.py`, `app/audio/vad.py` |
| 2.2 Drift monitoring | `app/logging_pipeline/feature_log.py`, `scripts/drift_check.py`, `scripts/build_drift_reference.py` |
| 2.3 Label-collection queue | `app/logging_pipeline/review_queue.py`, `/admin/review-queue/*` endpoints |
| 2.4 Privacy/consent | `app/logging_pipeline/consent.py`, `docs/CONSENT_RETENTION_POLICY.md` |
| 2.5 Alerting | `app/alerts/webhook.py`, wired into `scripts/drift_check.py` |
| 3.1 Model registry | HF Hub repo + `docs/MODEL_CARD_TEMPLATE.md`, `app/model/registry.py` |
| 3.2 Rollout (canary/blue-green/shadow) | `app/routing/canary.py`, `docs/ROLLOUT.md` |
| 3.3 Session-level state | `app/session/state.py` |
| 3.4 Latency | `scripts/benchmark_latency.py`, tracked in `app/inference/predict.py`'s `latency_ms` |
| 3.5 CI/CD | `.github/workflows/ci-eval-gate.yml` |


---
title: Ser Inference
emoji: 🚀
colorFrom: pink
colorTo: green
sdk: docker
pinned: false
---

