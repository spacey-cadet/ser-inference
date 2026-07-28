"""
Minimal model registry.

The "real" registry is the HF Hub repo itself: each retrain gets pushed as a
new revision/tag, with a model card recording training data version, which
retrain cycle it came from, hyperparameters, and per-corpus eval metrics
(RAVDESS / IEMOCAP / CREMA-D / chat-domain holdout) plus latency numbers.
See docs/MODEL_CARD_TEMPLATE.md.

This module just holds whichever revisions are currently loaded in this
process — the champion always, and optionally a challenger for canary
traffic. A self-hosted MLflow + SQLite instance is a drop-in free upgrade if
you outgrow "the Hub repo is my registry" — swap this module's internals,
keep the interface.
"""
from dataclasses import dataclass
from typing import Optional

from app.model.loader import LoadedModel, load_model, verify_model


@dataclass
class ModelRegistry:
    champion: LoadedModel
    challenger: Optional[LoadedModel] = None

    @classmethod
    def bootstrap(cls, settings) -> "ModelRegistry":
        champion = load_model(
            model_id=settings.model_id,
            hf_token=settings.hf_token,
            cache_dir=settings.cache_dir,
            revision="main",
        )
        verify_model(champion)

        challenger = None
        if settings.challenger_model_id and settings.canary_pct > 0:
            challenger = load_model(
                model_id=settings.challenger_model_id,
                hf_token=settings.hf_token,
                cache_dir=settings.cache_dir,
                revision=settings.challenger_revision,
            )
            verify_model(challenger)
            print(f"Challenger loaded — routing {settings.canary_pct}% of traffic to it")

        return cls(champion=champion, challenger=challenger)
