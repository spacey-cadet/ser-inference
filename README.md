# ser-inference

Speech emotion recognition service — WavLM backbone (frozen) + a small
trainable classifier head, built for use inside a chat app: decode →
VAD → embed → calibrate → cascade → session-aware emotion state.

Two deployment targets exist for this repo, sharing the same model and
serving code:

| Branch | Target | Cost | Status |
|---|---|---|---|
| `main` | HF Spaces / local docker (self-hostable) | free | original target |
| `aws-serverless` | AWS Lambda + S3 + DynamoDB, Kaggle-orchestrated retraining | ≤ $5/month | build in progress |

Only `infra/`-level deployment config and a handful of pluggable
storage backends differ between them — model code, calibration,
cascade logic, and consent handling are identical on both branches by
design.

## Where to start

- **Understand the inference pipeline itself** (VAD, calibration,
  cascade, drift, champion/challenger, review queue, consent):
  [`PIPELINE_README.md`](./PIPELINE_README.md). Read this first — it's
  the source of truth for what each `app/` module actually does.
- **Deploying to AWS**: start with
  [`infra/aws-serverless/README.md`](./infra/aws-serverless/README.md),
  then [`infra/aws-serverless/SETUP_ORDER.md`](./infra/aws-serverless/SETUP_ORDER.md)
  for the actual dependency-ordered setup sequence. The full design
  rationale (why each AWS service was picked, cost breakdown, what was
  deliberately left out) is in
  [`docs/decisions/project3-ser-aws-execution-plan.md`](./docs/decisions/project3-ser-aws-execution-plan.md).
- **Retraining loop internals** (policy-gated triggering, Kaggle
  orchestration, the champion/challenger promotion gate): see
  `scripts/check_retrain_policy.py`, `scripts/promotion_gate.py`, and
  `kaggle/train_head_kernel.py` — each has a docstring explaining its
  role in the loop, not just its arguments.

## Architecture, one level down

```
audio in ─▶ decode ─▶ VAD ─▶ WavLM (frozen) ─▶ MLP head ─▶ calibration
                                                              │
                                                        cascade + session state
                                                              │
                                                        prediction out
```

Only the MLP head is trainable. That single fact is why retraining
doesn't need a GPU in the routine case — see the AWS README for why
that matters for cost.

## Human-in-the-loop data collection

The model's training data is otherwise static. Low-confidence,
consented predictions get queued for a human to label
(`app/logging_pipeline/review_queue.py`, `consent.py`); labeled batches
periodically retrain the classifier head, gated by an explicit
promotion policy rather than auto-deployed. This closes the loop
without needing to re-collect a dataset from scratch — see
`docs/decisions/project3-ser-aws-execution-plan.md` for the full
data-flow diagram on the AWS branch specifically.

## Evaluation philosophy

Aggregate accuracy flatters easy classes and hides the confusions that
actually matter in a chat context (e.g. Sad/Neutral). `scripts/eval_report.py`
reports PR-AUC and per-class F1/recall alongside accuracy — that's what
to read in any eval, not the headline accuracy number. It's a pure
reporting step by design; `scripts/promotion_gate.py` is the separate
piece that turns two of its reports (champion vs. challenger, scored
against an identical pinned held-out set) into an actual promote/reject
decision, currently gated on macro F1 with a 0.01 minimum improvement
and a per-class recall floor.

## Repo layout

```
app/                  inference pipeline — shared across both branches
  model/               loader.py, classifier.py
  logging_pipeline/     feature_log.py, review_queue.py, consent.py
  session/              state.py
  routing/               canary.py (champion/challenger routing)
scripts/               eval_report.py, champion_challenger_eval.py,
                        calibration_fit.py, drift_check.py,
                        promotion_gate.py, check_retrain_policy.py,
                        assemble_kaggle_batch.py, trigger_kaggle_kernel.py
kaggle/                training kernel + metadata (AWS branch only)
infra/aws-serverless/   Terraform, Dockerfile, deployment docs (AWS branch only)
alerts/                webhook.py
docs/decisions/         ADRs — what changed between branches and why,
                        not just the vendor-name swap
.github/workflows/     ci-eval-gate.yml, drift-check.yml,
                        retrain-policy-check.yml, retrain.yml,
                        deploy-aws-serverless.yml
```

## Contributing to either branch

Changes to `app/model/`, `app/logging_pipeline/consent.py`, or anything
in the core inference path should work identically on both branches —
if a change only makes sense on one, it belongs in `infra/` or a
branch-specific backend adapter, not in shared code. See
`app/logging_pipeline/README_INTEGRATION.md`-style notes in the AWS
docs for how the pluggable storage backends are meant to be added
without touching shared logic.