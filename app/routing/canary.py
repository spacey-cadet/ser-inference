"""
In-process canary traffic split (Track 3.2).

Both champion and challenger checkpoints are loaded in the same FastAPI
process; a random percentage of requests routes to the challenger. Step the
percentage up manually (10 -> 25 -> 50 -> 100) via the CANARY_PCT env var
after checking logged confidence/error signals; set it back to 0 on any
anomaly. No new infra — this is what a real load-balancer-level canary
approximates for a single-Space deployment.
"""
import random

from app.model.registry import ModelRegistry


def select_model(registry: ModelRegistry, canary_pct: float):
    """Returns (LoadedModel, is_challenger: bool)."""
    if registry.challenger is not None and random.random() * 100 < canary_pct:
        return registry.challenger, True
    return registry.champion, False
