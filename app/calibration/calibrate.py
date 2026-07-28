"""
Applies a calibration mapping fit offline (Track 1.2) to raw softmax scores.

The fitting itself (Platt scaling / isotonic regression on a held-out
validation fold, sklearn.calibration.calibration_curve for the reliability
diagram) happens in scripts/calibration_fit.py — that's an offline, notebook-
style job, not something run per-request. This module just loads whatever
that job produced (data/calibration/calibration.json) and applies it.

If no calibration file exists yet, this is a no-op passthrough so the service
still runs before Phase 1 calibration work is done — but the API surfaces
`"calibrated": false` in the response so it's visible, not silently assumed.
"""
import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CalibrationMap:
    method: str  # "platt" | "isotonic" | "none"
    # Platt: per-class logistic (a, b) such that p = sigmoid(a*x + b)
    platt_params: Optional[dict] = None
    # Isotonic: per-class list of (x, y) breakpoints for interpolation
    isotonic_points: Optional[dict] = None

    @classmethod
    def load(cls, path: str) -> "CalibrationMap":
        if not path or not os.path.exists(path):
            return cls(method="none")
        with open(path) as f:
            raw = json.load(f)
        return cls(
            method=raw.get("method", "none"),
            platt_params=raw.get("platt_params"),
            isotonic_points=raw.get("isotonic_points"),
        )

    def apply(self, probs: list[float]) -> list[float]:
        if self.method == "none":
            return probs
        if self.method == "platt":
            return self._apply_platt(probs)
        if self.method == "isotonic":
            return self._apply_isotonic(probs)
        return probs

    def _apply_platt(self, probs: list[float]) -> list[float]:
        out = []
        for i, p in enumerate(probs):
            a, b = self.platt_params[str(i)]
            logit = np.log(p / (1 - p + 1e-9) + 1e-9)
            calibrated = 1 / (1 + np.exp(-(a * logit + b)))
            out.append(float(calibrated))
        total = sum(out) or 1.0
        return [v / total for v in out]

    def _apply_isotonic(self, probs: list[float]) -> list[float]:
        out = []
        for i, p in enumerate(probs):
            xs, ys = zip(*self.isotonic_points[str(i)])
            calibrated = float(np.interp(p, xs, ys))
            out.append(calibrated)
        total = sum(out) or 1.0
        return [v / total for v in out]
