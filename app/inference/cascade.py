"""
Two-threshold confidence cascade (Track 1.3), chat-safe.

    below threshold_low        -> "neutral" / no-signal for the response
                                   system, but still logged for review
    threshold_low..threshold_high -> use the label, keep a confidence tag
                                   attached downstream
    above threshold_high       -> use normally

The point: a confidently wrong label changes the tone of a reply in a way a
user notices. An honestly-uncertain signal costs nothing if the downstream
response logic treats it as "no adjustment."
"""
from dataclasses import dataclass
from enum import Enum


class CascadeTier(str, Enum):
    NO_SIGNAL = "no_signal"       # below threshold_low
    TENTATIVE = "tentative"       # between thresholds
    CONFIDENT = "confident"       # above threshold_high


@dataclass
class CascadeResult:
    tier: CascadeTier
    top_emotion: str
    top_score: float
    ranking: list[dict]
    response_emotion: str  # what the downstream response system should actually use
    log_for_review: bool   # Track 2.3 — feeds the label-collection queue


def apply_cascade(ranking: list[dict], threshold_low: float, threshold_high: float) -> CascadeResult:
    """
    ranking: list of {"emotion": str, "score": float}, sorted descending by score.
    """
    top = ranking[0]
    top_emotion, top_score = top["emotion"], top["score"]

    if top_score < threshold_low:
        tier = CascadeTier.NO_SIGNAL
        response_emotion = "neutral"
        log_for_review = True
    elif top_score < threshold_high:
        tier = CascadeTier.TENTATIVE
        response_emotion = top_emotion
        log_for_review = True
    else:
        tier = CascadeTier.CONFIDENT
        response_emotion = top_emotion
        log_for_review = False

    return CascadeResult(
        tier=tier,
        top_emotion=top_emotion,
        top_score=top_score,
        ranking=ranking,
        response_emotion=response_emotion,
        log_for_review=log_for_review,
    )
