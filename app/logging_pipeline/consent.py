"""
Consent gate for logging real user audio (Track 2.4).

This is a policy decision made before the first log write, not a technical
control after the fact — see docs/CONSENT_RETENTION_POLICY.md for the actual
policy text to adapt and publish.

What this module does:
  - Gives the API layer one place to check "am I allowed to log this
    request's features / keep a reference to this audio" based on a consent
    flag the client is expected to send.
  - Strips anything that could identify a user out of the feature vector
    before it's persisted — the feature vector itself (duration, RMS,
    centroid, pitch) is already non-identifying, but this is the seam where
    you'd add stripping if you ever log more (e.g. raw audio references).

If CONSENT_REQUIRED=true and no consent flag is present, the API still runs
inference and returns the emotion label to the response system (the product
needs that to function) but skips writing anything to the feature log /
review queue.
"""
from dataclasses import dataclass


@dataclass
class ConsentDecision:
    can_log: bool
    reason: str


def evaluate_consent(consent_flag_from_client: bool | None, consent_required: bool) -> ConsentDecision:
    if not consent_required:
        return ConsentDecision(can_log=True, reason="consent not required by config")
    if consent_flag_from_client:
        return ConsentDecision(can_log=True, reason="client provided consent")
    return ConsentDecision(can_log=False, reason="no consent flag from client; logging skipped")


def strip_identifying_metadata(session_id: str | None) -> str | None:
    """Placeholder seam for anonymization. Today session_id is expected to
    already be an opaque client-generated ID, not a real user identifier —
    enforce that at the client integration layer. If you ever start logging
    raw audio references, hash/tokenize any user-supplied filename here
    before it's persisted."""
    return session_id
