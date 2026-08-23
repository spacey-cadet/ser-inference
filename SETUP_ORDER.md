# Setup order — this stuff has real dependencies, don't run it out of order

## 1. Kaggle side (one-time)
- [ ] Create the pinned eval-set dataset (`ser-eval-set-pinned`) — do this
      FIRST and never change it, or champion/challenger comparisons stop
      being meaningful.
- [ ] Create an initial `ser-embeddings-cache` dataset (can be empty/seed).
- [ ] Create `ser-retrain-batch` dataset (empty placeholder — `assemble_kaggle_batch.py`
      versions it, doesn't create it from scratch).
- [ ] Fill in `kaggle/kernel-metadata.json`'s real dataset slugs and your
      Kaggle username.
- [ ] Confirm exact torch/transformers versions your Kaggle env uses —
      these MUST match `requirements.txt` used in `Dockerfile.lambda`,
      or CPU-serving numbers won't match what you saw in Kaggle.

## 2. HF Hub side (one-time)
- [ ] Push the current Kaggle-trained checkpoint as the initial champion
      revision, including a `metrics.json` (run `eval_report.py` against
      it on the pinned eval set first, since nothing has that yet).
- [ ] Note the exact WavLM backbone revision used — pin it in both
      `kaggle/train_head_kernel.py` and `app/model/loader.py`.

## 3. AWS side
- [ ] `terraform apply` on `github_oidc.tf` FIRST (nothing else depends
      on it, and the GH Actions workflows need `AWS_DEPLOY_ROLE_ARN`
      before they can run at all).
- [ ] Set GitHub secrets: `AWS_DEPLOY_ROLE_ARN`, `KAGGLE_USERNAME`,
      `KAGGLE_KEY`, `HF_TOKEN`, `ALERT_WEBHOOK_URL`.
- [ ] Set GitHub variables (not secrets — these aren't sensitive):
      `HF_MODEL_REPO`, `HF_CHAMPION_REVISION`.
- [ ] `terraform apply` on the rest (`main.tf`, `dynamodb.tf`, `s3.tf`,
      `monitoring.tf`) — this creates an ECR repo with nothing in it yet,
      so the Lambda function resource will fail until step 4.

## 4. First deploy
- [ ] Run `deploy-aws-serverless.yml` manually (`workflow_dispatch`) to
      build+push the first image and get the Lambda working.
- [ ] Confirm the Function URL responds to a test `/predict` call.
- [ ] Seed `ser-inference-retrain-state` table with a `watermark` item
      (`last_trained_at` = now) so the first policy check doesn't try to
      pull in old data as "new."

## 5. Only then — turn on the loop
- [ ] Enable `retrain-policy-check.yml`'s schedule.
- [ ] Watch the first real `retrain.yml` run closely — check
      `decision.json` and the webhook alert manually before trusting it
      to run unattended.

## Known open item this doesn't solve
`assemble_kaggle_batch.py`'s watermark update in `retrain.yml` advances
`last_trained_at` on BOTH promote and reject outcomes. That means a
rejected batch's labels aren't automatically re-included in the next
attempt's "new since watermark" count — they were used, just not
promoted. If you'd rather a reject leave those items eligible for
re-inclusion (e.g. combined with more new labels next time), change the
watermark update to only fire on promotion, and instead track "already
attempted" via `deployed`-vs-`labeled` status only. Worth deciding
deliberately rather than leaving as the current default.
