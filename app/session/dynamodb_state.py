"""
DynamoDB-backed session state.

The class implements the SessionStore get/update interface from
app.session.state, storing the same SessionEmotionState payload as JSON.
"""
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import boto3

from app.session.state import SessionEmotionState, SessionStore


class DynamoDBSessionStore(SessionStore):
    def __init__(self, table_name: str, ttl_minutes: int = 60):
        if not table_name:
            raise ValueError("SESSION_STATE_TABLE must be set when STORAGE_BACKEND=aws")
        self.table = boto3.resource("dynamodb").Table(table_name)
        self.ttl_minutes = ttl_minutes

    def get(self, session_id: str) -> Optional[SessionEmotionState]:
        resp = self.table.get_item(Key={"session_id": session_id})
        item = resp.get("Item")
        if not item:
            return None
        return SessionEmotionState.from_dict(json.loads(item["state_json"]))

    def update(self, session_id: str, emotion: str, score: float, decay: float, history_len: int) -> SessionEmotionState:
        state = self.get(session_id) or SessionEmotionState(session_id, {}, [])
        for e in list(state.decayed_scores.keys()):
            state.decayed_scores[e] *= 1 - decay
        state.decayed_scores[emotion] = state.decayed_scores.get(emotion, 0.0) + decay * score
        state.history.append({"emotion": emotion, "score": score, "ts": time.time()})
        state.history = state.history[-history_len:]
        state.updated_at = time.time()

        now = datetime.now(timezone.utc)
        expires = now + timedelta(minutes=self.ttl_minutes)
        self.table.put_item(
            Item={
                "session_id": session_id,
                "state_json": json.dumps(state.to_dict()),
                "updated_at": now.isoformat(),
                "expires_at": int(expires.timestamp()),
            }
        )
        return state
