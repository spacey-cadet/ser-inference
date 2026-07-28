# Architecture

## Request flow (`POST /predict`)

```
                          ┌─────────────────────────┐
  audio file, session_id, │   FastAPI /predict       │
  consent flag  ────────► │   (app/api/routes.py)   │
                          └────────────┬─────────────┘
                                       │
                          ┌────────────▼─────────────┐
                          │ Canary routing            │  Track 3.2
                          │ (app/routing/canary.py)   │  champion or challenger
                          └────────────┬─────────────┘
                                       │
                          ┌────────────▼─────────────┐
                          │ decode + resample          │  app/audio/decode.py
                          │ (PyAV, any codec → 16k mono)│
                          └────────────┬─────────────┘
                                       │
                          ┌────────────▼─────────────┐
                          │ validation gate             │  Track 2.1
                          │ duration / RMS silence floor │  app/audio/validate.py
                          └────────────┬─────────────┘
                                  reject│  pass
                       ┌───────────────┘   │
                       │                   ▼
                       │      ┌─────────────────────────┐
                       │      │ VAD trim (silero)         │  Track 2.1
                       │      │ app/audio/vad.py          │
                       │      └────────────┬─────────────┘
                       │                   │
                       │      ┌────────────▼─────────────┐
                       │      │ WavLM + pooling + MLP      │  app/inference/predict.py
                       │      │ classifier forward pass    │  (run_inference)
                       │      └────────────┬─────────────┘
                       │                   │
                       │      ┌────────────▼─────────────┐
                       │      │ calibration (Platt/isotonic)│ Track 1.2
                       │      │ app/calibration/calibrate.py│
                       │      └────────────┬─────────────┘
                       │                   │
                       │      ┌────────────▼─────────────┐
                       │      │ confidence cascade           │ Track 1.3
                       │      │ app/inference/cascade.py     │
                       │      └────────────┬─────────────┘
                       │                   │
                       ▼                   ▼
             response_emotion=neutral   response_emotion=<label or neutral>
             (chat-safe default)        cascade_tier attached
                       │                   │
                       │      ┌────────────▼─────────────┐
                       │      │ session state update        │ Track 3.3
                       │      │ app/session/state.py        │
                       │      └────────────┬─────────────┘
                       │                   │
                       └──────►┌───────────▼─────────────┐
                                │ consent check                │ Track 2.4
                                │ app/logging_pipeline/consent │
                                └────────────┬─────────────┘
                                        can_log │ can't log
                                             ▼        │
                                ┌─────────────────────┐│
                                │ feature/drift log     ││ Track 2.2/2.3
                                │ SQLite or HF Dataset   ││
                                └─────────────┬────────┘│
                                              │◄─────────┘
                                              ▼
                                    JSON response to caller
```

## Offline / scheduled paths (not on the request critical path)

```
scripts/calibration_fit.py  ──► data/calibration/calibration.json  ──► loaded at app startup
scripts/build_drift_reference.py ──► data/drift_reference/train_features.json
.github/workflows/drift-check.yml (cron) ──► scripts/drift_check.py
    reads live features from feature log, compares to reference via ks_2samp,
    fires app/alerts/webhook.py on drift or high low-confidence rate
scripts/champion_challenger_eval.py ──► promotion recommendation across all corpora
.github/workflows/ci-eval-gate.yml ──► tests + eval gate on every merge to main
```

## Component responsibilities

- **`app/model/`** — loading and sanity-checking checkpoints from HF Hub
  revisions. `registry.py` is the seam where a self-hosted MLflow instance
  could replace "the Hub repo is my registry" later without touching
  anything else.
- **`app/audio/`** — everything that happens to a clip before it reaches the
  model. Fail-safe by design: VAD fails open (scores raw audio if unavailable
  rather than crashing), validation fails closed (rejects and returns
  neutral rather than scoring garbage).
- **`app/calibration/` + `app/inference/`** — the actual modeling-rigor
  layer. Calibration is a pure function of an offline-fit JSON file; the
  cascade is pure logic with no I/O — both are unit-testable without a model.
- **`app/session/`** — the one piece of state that spans requests. Kept
  behind a `SessionStore` interface specifically so swapping memory → Redis
  is a config change, not a rewrite.
- **`app/logging_pipeline/`** — the data flywheel. `consent.py` sits in
  front of everything else in this package on purpose — nothing else in the
  package should be reachable from the request path without going through it
  first.
- **`app/routing/`** — single-purpose: pick champion or challenger. Kept
  separate from `predict.py` so the canary logic can be tested/changed
  without touching the inference pipeline.
- **`app/alerts/`** — intentionally the simplest module in the repo. Two
  lines of actual logic. Resist the urge to grow it in place — an on-call
  paging system is a different tool, not a bigger webhook module.
