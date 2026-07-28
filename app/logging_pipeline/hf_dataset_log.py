"""
Append-only private Hugging Face Dataset backend for request/feature logging.

Use this instead of SqliteFeatureLog when the Space doesn't have persistent
disk storage (logs would vanish on every restart) — a private HF Dataset
repo survives redeploys and is still free.

This buffers rows locally and flushes in batches via `upload_file`, since the
Hub API isn't meant for per-row appends. A scheduled GitHub Action or a
periodic in-process flush (see app/api/routes.py startup/shutdown hooks) is
the intended flush trigger.
"""
import json
import time
from pathlib import Path


class HFDatasetFeatureLog:
    def __init__(self, repo_id: str, hf_token: str, buffer_path: str = "./data/hf_dataset_buffer.jsonl"):
        if not repo_id:
            raise ValueError("HF_DATASET_REPO must be set to use the hf_dataset backend")
        self.repo_id = repo_id
        self.hf_token = hf_token
        self.buffer_path = Path(buffer_path)
        self.buffer_path.parent.mkdir(parents=True, exist_ok=True)

    def log_request(self, request_id, session_id, features, top_emotion, top_score,
                     cascade_tier, latency_ms, model_revision, consent_granted):
        row = {
            "request_id": request_id, "ts": time.time(), "session_id": session_id,
            **features, "top_emotion": top_emotion, "top_score": top_score,
            "cascade_tier": cascade_tier, "latency_ms": latency_ms,
            "model_revision": model_revision, "consent_granted": consent_granted,
        }
        with open(self.buffer_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    def flush(self):
        """Push the buffered JSONL file to the HF Dataset repo. Call this on a
        schedule (e.g. every N requests, or via a GitHub Action) rather than
        every request."""
        from huggingface_hub import HfApi
        if not self.buffer_path.exists() or self.buffer_path.stat().st_size == 0:
            return
        api = HfApi(token=self.hf_token)
        remote_name = f"logs/{int(time.time())}.jsonl"
        api.upload_file(
            path_or_fileobj=str(self.buffer_path),
            path_in_repo=remote_name,
            repo_id=self.repo_id,
            repo_type="dataset",
        )
        self.buffer_path.unlink()
