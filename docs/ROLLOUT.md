# Rollout Playbook

Three approximations of real rollout mechanisms, all buildable on free-tier
infrastructure. Each is a genuine approximation of a specific real
mechanism — worth knowing what the paid version adds so you're not
surprised later.

## 1. In-process canary (default recommendation for a single Space)

**Mechanism:** both champion and challenger checkpoints loaded in the same
FastAPI process; a random percentage of requests routes to the challenger.

**How to use it:**
1. Push the new checkpoint to the Hub under a new revision/tag.
2. Set in `.env` (or Space secrets):
   ```
   CHALLENGER_MODEL_ID=<repo>          # usually same as MODEL_ID
   CHALLENGER_REVISION=<new tag>
   CANARY_PCT=10
   ```
3. Restart the Space. `app/model/registry.py` loads both; `app/routing/canary.py`
   routes `CANARY_PCT`% of traffic to the challenger.
4. Every `/predict` response includes `"is_challenger": true/false` and
   `"model_revision"` — filter your feature log / dashboards on this to
   compare champion vs. challenger outcomes on identical live traffic.
5. Step up manually: 10 → 25 → 50 → 100, checking confidence distribution
   and cascade-tier mix at each step. Drop back to `CANARY_PCT=0`
   immediately on any anomaly (confidence collapse, latency spike, error
   rate).
6. Once at 100% and stable for a chosen soak period, promote: set
   `MODEL_ID`'s default revision or `CHALLENGER_REVISION` as the new
   `main`, drop `CHALLENGER_MODEL_ID`, redeploy.

**What a real load balancer adds:** true percentage-based routing at the
network layer (not per-process randomness), automatic SLA-triggered
rollback without a human watching dashboards, and routing that doesn't
require both models resident in the same process's memory (relevant once a
challenger checkpoint is large enough that loading two at once strains the
free tier's RAM).

## 2. Blue-green (two Spaces)

**Mechanism:** a staging Space and a production Space; cut over by pointing
consumers at the new Space's URL; rollback is pointing back.

**How to use it:**
1. Duplicate the Space (HF Spaces supports this natively) as `<name>-staging`.
2. Deploy the challenger to staging, run your test suite (`pytest tests/`)
   and a manual smoke test against it directly.
3. Update the chat application's configured inference URL to the staging
   Space once satisfied.
4. Keep the old production Space running, untouched, for a rollback window
   — pointing the client config back is the entire rollback procedure.

**What a real blue-green setup adds:** automated traffic cutover (DNS/load
balancer level, not a client config change), health-check-gated automatic
cutover, and zero-downtime cutover for high-QPS services (a client config
change has a brief window where in-flight requests may hit the old URL
during redeploy).

## 3. Shadow deployment

**Mechanism:** a second free Space running the challenger, mirroring a
sample of real requests, logging output without ever returning it to the
user. Useful specifically for validating latency under real chat traffic
before a challenger goes anywhere near production.

**How to use it:**
1. Deploy the challenger to a separate Space (`<name>-shadow`).
2. In the chat application (not this repo — this is a client-side
   integration), fire a sampled fraction of real requests to both the
   production Space and the shadow Space, but only use the production
   Space's response.
3. Compare latency and confidence-tier distributions between the two logs.
   This is the cleanest way to validate `scripts/benchmark_latency.py`-style
   numbers under real traffic shape (codec mix, duration distribution)
   rather than synthetic sample clips.

**What this doesn't give you:** shadow traffic never influences what the
user sees, so it can't validate anything about response-system behavior —
only model-level latency and prediction distribution.

## Choosing between them

- **In-process canary** — cheapest, fastest to iterate, good default for
  comparing model quality on live traffic. Downside: both checkpoints share
  the same process's CPU/RAM.
- **Blue-green** — best when you want a hard, instant rollback path and
  don't need gradual traffic ramp.
- **Shadow** — best for latency/capacity validation before a challenger is
  trusted with any live traffic at all, especially after an architecture
  change (quantization, ONNX export, trimmed layers per Track 3.4).

In practice: shadow first (validate latency doesn't regress) → in-process
canary (validate prediction quality on live traffic, ramp gradually) →
blue-green as the final cutover once canary is at 100% and stable.
