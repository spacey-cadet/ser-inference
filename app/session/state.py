"""
Session-keyed rolling emotion state (Track 3.3).

A chat is multi-turn, so a decayed average / short history of the last N
turns' emotion labels, keyed by session or user ID, is a legitimate,
low-effort feature store use case — even though the classifier itself is
single-utterance.

Two backends:
  - memory: a process-local dict. Fine for a single free Space instance.
    State resets on restart/redeploy — acceptable for a first pass.
  - redis: Upstash free tier or any Redis URL. Use once you have >1 replica
    or want state to survive restarts.

Not attempted here: feeding this state back into training data. The guide
flags that as the point where online/offline computation divergence becomes
a real risk — worth solving deliberately later, not by accident now.
"""
import json
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SessionEmotionState:
    session_id: str
    decayed_scores: dict  # emotion -> decayed running score
    history: list  # list of {"emotion": str, "score": float, "ts": float}, capped
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "decayed_scores": self.decayed_scores,
            "history": self.history,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SessionEmotionState":
        return cls(**d)

    @property
    def dominant_emotion(self) -> Optional[str]:
        if not self.decayed_scores:
            return None
        return max(self.decayed_scores, key=self.decayed_scores.get)


class SessionStore:
    """Interface both backends implement: get / update."""

    def get(self, session_id: str) -> Optional[SessionEmotionState]:
        raise NotImplementedError

    def update(self, session_id: str, emotion: str, score: float, decay: float, history_len: int) -> SessionEmotionState:
        raise NotImplementedError


class InMemorySessionStore(SessionStore):
    def __init__(self):
        self._store: dict[str, SessionEmotionState] = {}

    def get(self, session_id: str) -> Optional[SessionEmotionState]:
        return self._store.get(session_id)

    def update(self, session_id: str, emotion: str, score: float, decay: float, history_len: int) -> SessionEmotionState:
        state = self._store.get(session_id) or SessionEmotionState(session_id, {}, [])
        for e in list(state.decayed_scores.keys()):
            state.decayed_scores[e] *= (1 - decay)
        state.decayed_scores[emotion] = state.decayed_scores.get(emotion, 0.0) + decay * score
        state.history.append({"emotion": emotion, "score": score, "ts": time.time()})
        state.history = state.history[-history_len:]
        state.updated_at = time.time()
        self._store[session_id] = state
        return state


class RedisSessionStore(SessionStore):
    def __init__(self, redis_url: str):
        import redis  # lazy import — only required if SESSION_BACKEND=redis
        self._client = redis.Redis.from_url(redis_url, decode_responses=True)

    def _key(self, session_id: str) -> str:
        return f"ser:session:{session_id}"

    def get(self, session_id: str) -> Optional[SessionEmotionState]:
        raw = self._client.get(self._key(session_id))
        if not raw:
            return None
        return SessionEmotionState.from_dict(json.loads(raw))

    def update(self, session_id: str, emotion: str, score: float, decay: float, history_len: int) -> SessionEmotionState:
        state = self.get(session_id) or SessionEmotionState(session_id, {}, [])
        for e in list(state.decayed_scores.keys()):
            state.decayed_scores[e] *= (1 - decay)
        state.decayed_scores[emotion] = state.decayed_scores.get(emotion, 0.0) + decay * score
        state.history.append({"emotion": emotion, "score": score, "ts": time.time()})
        state.history = state.history[-history_len:]
        state.updated_at = time.time()
        self._client.set(self._key(session_id), json.dumps(state.to_dict()), ex=60 * 60 * 24)  # 24h TTL
        return state


def build_session_store(settings) -> SessionStore:
    if settings.session_backend == "redis" and settings.redis_url:
        return RedisSessionStore(settings.redis_url)
    return InMemorySessionStore()
