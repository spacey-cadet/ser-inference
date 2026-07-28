"""
Loads a WavLM + StatisticsPooling + MLPClassifier stack from a Hugging Face
Hub model repo revision.

This is deliberately revision-aware (not just repo-aware) so the same function
loads either the champion or a challenger checkpoint — the mechanism behind
the in-process canary and the offline champion/challenger comparison
(Track 1.5, Track 3.2).
"""
import os
from pathlib import Path

import torch

from app.model.classifier import MLPClassifier

WAVLM_SOURCE = "microsoft/wavlm-base-plus"


class LoadedModel:
    def __init__(self, wavlm, pooling, classifier, revision_tag: str):
        self.wavlm = wavlm
        self.pooling = pooling
        self.classifier = classifier
        self.revision_tag = revision_tag  # e.g. "champion@main" or "challenger@v2"


def load_model(model_id: str, hf_token: str, cache_dir: str, revision: str = "main") -> LoadedModel:
    from huggingface_hub import snapshot_download
    from speechbrain.lobes.models.huggingface_transformers.wavlm import WavLM
    from speechbrain.nnet.pooling import StatisticsPooling

    local_dir = snapshot_download(
        repo_id=model_id,
        repo_type="model",
        revision=revision,
        token=hf_token or None,
        local_dir=os.path.join(cache_dir, revision.replace("/", "_")),
        ignore_patterns=["pretrained_models/**"],
    )

    ckpt_dirs = sorted(Path(local_dir).glob("CKPT+*"))
    if not ckpt_dirs:
        raise FileNotFoundError(f"No CKPT+* folder found in {local_dir}")
    ckpt_dir = ckpt_dirs[-1]

    wavlm = WavLM(
        source=WAVLM_SOURCE,
        save_path=os.path.join(cache_dir, "pretrained_models/wavlm"),
        output_norm=True,
        freeze=True,
        freeze_feature_extractor=True,
    )

    wavlm_ckpt = ckpt_dir / "wavlm.ckpt"
    if wavlm_ckpt.exists():
        state = torch.load(str(wavlm_ckpt), map_location="cpu", weights_only=True)
        wavlm.load_state_dict(state, strict=False)

    pooling = StatisticsPooling()
    classifier = MLPClassifier()

    clf_ckpt = ckpt_dir / "classifier.ckpt"
    if not clf_ckpt.exists():
        raise FileNotFoundError(f"classifier.ckpt not found in {ckpt_dir}")
    state = torch.load(str(clf_ckpt), map_location="cpu", weights_only=True)
    classifier.load_state_dict(state, strict=True)

    wavlm.eval()
    pooling.eval()
    classifier.eval()

    return LoadedModel(wavlm, pooling, classifier, revision_tag=f"{model_id}@{revision}")


def verify_model(model: LoadedModel):
    """Sanity check — a trained model should not produce ~uniform 1/8 probabilities."""
    dummy = torch.randn(1, 16000)
    with torch.no_grad():
        feat = model.wavlm(dummy)
        if isinstance(feat, dict):
            feat = feat["last_hidden_state"]
        pooled = model.pooling(feat, torch.tensor([1.0]))
        logits = model.classifier(pooled)
        probs = torch.softmax(logits, dim=-1).squeeze(0)

    max_conf = probs.max().item()
    if max_conf < 0.2:
        raise RuntimeError(
            f"[{model.revision_tag}] looks uninitialised (max_conf={max_conf:.3f}). "
            "Check that classifier.ckpt keys match MLPClassifier."
        )
    print(f"[{model.revision_tag}] sanity check OK — max_conf={max_conf:.3f}")
