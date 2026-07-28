# Consent & Retention Policy (Template)

This is a policy document, not code — fill in the bracketed parts for your
product and publish it (ToS update, privacy page, or in-app disclosure)
**before** `FEATURE_LOG_BACKEND` is turned on for real user traffic. The
`consent` form field on `/predict` is meaningless if there's nothing behind
it for a user to have actually consented to.

## What is logged

When a user's client sends `consent=true` on a request, this pipeline logs,
per request:

- A small acoustic feature vector: clip duration, RMS energy, spectral
  centroid, a rough pitch estimate.
- The predicted emotion label, its confidence score, and which confidence
  tier it fell into (confident / tentative / no-signal).
- Inference latency and which model revision served the request.
- An opaque `session_id` if the client supplied one (this must be a
  client-generated identifier, not a real name/email/account ID — enforce
  this at the client integration layer, not here).

**Raw audio is not logged or stored by default.** The feature vector above
is derived from the audio and does not reconstruct it. If a future version
of this pipeline starts storing audio references (e.g. to support richer
manual review), this policy must be updated first, and that storage should
go through the anonymization seam in `app/logging_pipeline/consent.py`
(`strip_identifying_metadata`) before persistence.

## Why it's logged

1. **Drift monitoring** (Track 2.2) — comparing live audio characteristics
   against the training distribution to catch when the model is operating
   further from its training data than expected.
2. **The retraining data loop** (Track 2.3) — low-confidence predictions are
   the highest-value candidates for the next round of manual labeling and
   model retraining.

## What consent covers

[Fill in: reference your product's actual ToS / privacy policy section
here. At minimum, this should disclose that voice input may be analyzed for
emotional tone, that derived (non-audio) features may be retained for
service-improvement purposes, and how a user can decline.]

## Retention

- Logged feature records are retained for **`RETENTION_DAYS`** days
  (default 30, set in `.env`) and then purged. `SqliteFeatureLog.purge_older_than`
  implements this for the SQLite backend — schedule it (e.g. as part of the
  drift-check GitHub Action) if you're using that backend continuously.
- For the `hf_dataset` backend, apply the same retention window manually by
  periodically pruning old batch files from the dataset repo, or by
  configuring the repo's own retention if the Hub adds that capability.

## Access control

- The raw review queue (`/admin/review-queue/export`) should sit behind
  authentication before this pipeline handles real user traffic — the
  scaffold does not add auth by default; add it at the FastAPI layer (e.g.
  an API key check dependency) or restrict the endpoint at the network
  level before going live.
- Whoever performs manual review of the low-confidence queue should be a
  named, limited set of people, consistent with who has access to any other
  user-generated content in the product.

## What happens if consent is not given

Inference still runs and the emotion label is still returned to the calling
application — the product needs that to function turn-by-turn. Nothing
about that request is written to the feature log or review queue. This is
enforced in `app/api/routes.py` via `evaluate_consent()` before any call to
`feature_log.log_request(...)`.

## Before turning logging on, confirm:

- [ ] ToS / privacy disclosure covering this use is published
- [ ] Client only sends `consent=true` after genuine user consent
- [ ] `RETENTION_DAYS` is set to a value your legal/product review has signed off on
- [ ] Access to `/admin/review-queue/*` is restricted
- [ ] Whoever reviews the queue has been told what they're looking at and why
