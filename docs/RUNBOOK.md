# Runbook

Free-tier equivalent of on-call docs. There's no paging here (see
`docs/PLAN.md`'s "what still needs paid infra") — this assumes a human sees
the Slack/Discord alert and acts.

## Alert: drift check fired (KS-test p < threshold)

1. Check `scripts/drift_check.py` output (or re-run it) — which feature(s)
   fired (`duration_sec`, `rms_energy`, `spectral_centroid`, `pitch_estimate`)?
2. If this is the **first** time it's fired after deploying: expected.
   Chat audio genuinely differs from RAVDESS/IEMOCAP/CREMA-D. Use it to
   sanity-check `MIN_DURATION_SEC` / `MAX_DURATION_SEC` / `RMS_SILENCE_FLOOR`
   are reasonable for what you're actually seeing, then move on.
3. If it's a **new** drift signal after being stable: check for a client
   change (new codec, new mic pipeline, new browser MediaRecorder config)
   before assuming a model problem.
4. No immediate action required — drift is a signal to prioritize the next
   retrain's chat-domain data collection, not to take the service down.

## Alert: low-confidence rate exceeds threshold

1. Check `/admin/review-queue/stats` — is the queue growing, and how fast?
2. Sample a few rows via `/admin/review-queue/export?limit=20` — are these
   genuinely ambiguous clips, or is something upstream broken (e.g. wrong
   audio format reaching the endpoint, VAD stripping actual speech)?
3. If it looks like a genuine model/data gap: this is exactly the queue
   Track 2.3 exists for — prioritize a review pass and note it for the next
   retrain.
4. If it looks like a pipeline bug (e.g. every request is low-confidence
   starting at a specific deploy time): check recent deploys, consider
   `CANARY_PCT=0` if a challenger was recently ramped up.

## Incident: challenger canary looks worse than champion

1. Set `CANARY_PCT=0` immediately (env var + redeploy, or Space
   restart if hot-reload isn't wired up).
2. Compare `is_challenger: true` vs `false` request logs for the affected
   window — confidence distribution, cascade tier mix, latency.
3. Do not delete the challenger checkpoint from the Hub — you'll want it
   for postmortem analysis via `scripts/champion_challenger_eval.py`.

## Incident: latency regression

1. Re-run `scripts/benchmark_latency.py` against the currently deployed
   revision to confirm — free-tier Spaces have variable CPU allocation, so
   check whether this is a one-off noisy measurement or persistent.
2. If persistent and tied to a recent model change: check whether the new
   checkpoint has more unfrozen/fine-tuned transformer layers than the
   previous one (see Track 3.4 in the guide — this is a direct latency
   lever).
3. Options in order of effort: (a) revert to prior checkpoint via canary
   rollback, (b) decouple emotion scoring from the response path (score
   async, apply the tag to the next turn), (c) ONNX export / quantization
   (bigger lift, do this as planned work, not mid-incident).

## Routine: retraining data pull

1. `GET /admin/review-queue/export?limit=<N>` — pull the current backlog.
2. Label externally (spreadsheet, labeling tool — this repo doesn't
   prescribe one).
3. Mark reviewed via `app.logging_pipeline.review_queue.mark_reviewed(sqlite_path, request_ids)`
   (or the equivalent for the hf_dataset backend — flag rows in the labeled
   export instead).
4. Fold labeled examples into the next training run, prioritizing them over
   a random sample — they're the examples the model already struggled with.
5. After retraining, run `scripts/champion_challenger_eval.py` across all
   corpora + the chat-domain holdout before touching `CANARY_PCT`.

## Routine: retention purge

Run periodically (weekly is reasonable) if using the SQLite backend:

```python
from app.logging_pipeline.feature_log import SqliteFeatureLog
from app.config import get_settings
settings = get_settings()
SqliteFeatureLog(settings.sqlite_path).purge_older_than(settings.retention_days)
```

Wire this into a scheduled GitHub Action alongside `drift-check.yml` if you
want it fully automated.
