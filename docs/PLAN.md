# Build Plan

Same phasing as the pipeline guide, mapped to this repo's files. Each phase
is a weekend-sized chunk and produces something runnable — you're not
blocked waiting for a later phase to see value from an earlier one.

## Phase 1 — Foundations

**Goal: calibrated, threshold-safe inference with a known latency number.**
Nothing here needs new infrastructure — it runs on data/checkpoints you
already have.

1. Export raw softmax scores + true labels for your existing validation
   fold into a CSV (`true_label, prob_0..prob_7`).
2. `python scripts/calibration_fit.py --predictions <csv> --method isotonic`
   → produces `data/calibration/calibration.json` and the pre/post
   reliability diagrams. Use the diagrams as your evidence artifact that
   calibration actually improved things.
3. Confirm the app picks it up: start the service, hit `/`, check
   `"calibrated": true` in the health response.
4. Tune `THRESHOLD_LOW` / `THRESHOLD_HIGH` in `.env` against your validation
   set's calibrated confidence distribution — the cascade logic itself
   (`app/inference/cascade.py`) doesn't need to change.
5. `python scripts/benchmark_latency.py --audio-dir <sample clips>` →
   record p50/p99 in `docs/MODEL_CARD_TEMPLATE.md`.
6. Run `scripts/baseline_classical_model.py` once for the "WavLM beats
   classical baseline by X points" number — good for a model card / writeup,
   not on the serving path.

**Exit criteria:** `/predict` returns calibrated scores, the cascade
correctly defaults low-confidence predictions to neutral, and you have a
real p50/p99 latency number.

## Phase 2 — Input & logging

**Goal: chat-safe audio validation, and the start of the retraining data loop.**

1. `app/audio/validate.py` and `app/audio/vad.py` are already wired into
   `predict_from_path` — tune `MIN_DURATION_SEC`, `MAX_DURATION_SEC`,
   `RMS_SILENCE_FLOOR` against real chat-audio samples if you have any yet,
   otherwise leave the defaults and revisit after Phase 3's drift check
   starts firing.
2. Choose a feature-log backend: `sqlite` if the Space has persistent
   storage, `hf_dataset` otherwise (`.env`: `FEATURE_LOG_BACKEND`,
   `HF_DATASET_REPO`).
3. **Before turning logging on**, write and publish
   `docs/CONSENT_RETENTION_POLICY.md` (template included — fill in the
   product-specific parts) and make sure your client sends `consent=true`
   only after real user consent/ToS disclosure is in place.
4. Build the drift reference once against your training set:
   `python scripts/build_drift_reference.py --data-csv <training labels csv>`.

**Exit criteria:** requests are validated and VAD-trimmed before scoring;
low-confidence requests are logged with review flags; consent policy is
published before any real user audio is logged.

## Phase 3 — Monitoring & alerting

**Goal: know when reality diverges from your training data, automatically.**

1. Enable `.github/workflows/drift-check.yml` — set repo secrets
   (`HF_TOKEN`, `ALERT_WEBHOOK_URL`) and variables (`HF_DATASET_REPO` if
   using that backend).
2. Set up a Slack or Discord incoming webhook, put the URL in
   `ALERT_WEBHOOK_URL`.
3. Expect the drift check to fire early and often — chat audio diverging
   from RAVDESS/IEMOCAP/CREMA-D is the expected gap, not a bug. Use early
   alerts to tune `MIN_DURATION_SEC` / `RMS_SILENCE_FLOOR` rather than
   silencing the alert.
4. Start a weekly (or whatever cadence fits) manual review pass on
   `/admin/review-queue/export` — export, label, track via
   `app/logging_pipeline/review_queue.mark_reviewed`.

**Exit criteria:** a Slack/Discord message fires automatically on real
drift or a spiking low-confidence rate; someone is reviewing the queue on
a fixed cadence.

## Phase 4 — Registry, rollout & session state

**Goal: safe promotion path for the next checkpoint, and multi-turn emotion context.**

1. Every checkpoint pushed to the Hub gets a filled-out model card
   (`docs/MODEL_CARD_TEMPLATE.md`) — training data version, retrain lineage,
   per-corpus metrics, latency.
2. Set `CHALLENGER_MODEL_ID` / `CHALLENGER_REVISION` / `CANARY_PCT` in
   `.env` to test a new checkpoint against a slice of real traffic (see
   `docs/ROLLOUT.md` for the full canary → blue-green → shadow playbook).
3. `SESSION_BACKEND=memory` works for a single Space instance out of the
   box; move to `redis` (Upstash free tier) once you have >1 replica or
   want state to survive restarts.
4. Enable `.github/workflows/ci-eval-gate.yml`'s actual threshold checks
   once you have corpus-by-corpus prediction CSVs to gate on (it's a
   placeholder until then — don't let it silently pass).

**Exit criteria:** a challenger checkpoint can be promoted through canary →
full rollout with a documented rollback path; session-level emotion state
is live for multi-turn chats.

## Ongoing — once enough labeled chat audio accumulates

- Pull the highest-value examples from the review queue
  (`/admin/review-queue/export`) into the next retrain set — these are the
  examples the model already struggled with, not a random sample.
- Run `scripts/champion_challenger_eval.py` across RAVDESS, IEMOCAP,
  CREMA-D, and the new chat-domain holdout. Promote only if the challenger
  clears the bar on all four.
- Re-run `scripts/benchmark_latency.py` after any architecture change, not
  just after accuracy changes.

## What this plan deliberately does not solve

See `docs/PLAN.md`'s companion section in the original guide — true
load-balancer-level traffic splitting with SLA-triggered rollback, on-call
paging with escalation, a real sub-10ms streaming feature store, fully
automated retraining triggered off drift alerts, and GPU inference at scale
all genuinely need paid infrastructure. This repo gets you a free,
reasonable approximation of each; it doesn't pretend to replace them.
