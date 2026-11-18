"""Behavior-cloned grasp parameters: a tiny MLP distilled from verified grasp episodes.

Imitation-learning step of the local-first handoff: `scripts/gen_grasp_data.py` collects
(object features -> verified grasp parameters) in randomized sim scenes, with the scripted
controller as the base policy and an optional LLM as the DAgger expert on its failures;
`scripts/train_bc_grasp.py` fits the MLP with torch and saves it as plain JSON. This module
is the runtime side: a numpy-only forward pass (same spirit as the exported ONNX reach
policy - no torch at inference), producing parameters in the exact format `scripted_grasp`
replays and `distill_grasp` records (src/so101_reach/agent/local_skills.py).

Features are computable from any `Detection` alone (sim ground truth or real DINO), so the
model transfers to the real pipeline unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from so101_reach.agent.local_skills import is_soft

DEFAULT_MODEL = Path(__file__).resolve().parents[3] / "data" / "il" / "bc_grasp.json"
FORCE_CLASSES = ("gentle", "medium", "firm")
ROUND_WORDS = ("ball", "sphere", "orange", "apple", "cylinder", "can", "bottle")
FEATURE_NAMES = ("l_mm", "w_mm", "h_mm", "aspect", "tallness", "stacked", "round", "soft")


def features_for(obj, tz: float) -> list[float]:
  """Feature vector for one measured object (metres in, mixed units out; standardized by the
  model's stored normalization). Must stay in sync with FEATURE_NAMES and the training data."""
  l, w, h = (float(v) for v in obj.size)
  name = obj.name.lower()
  return [1000 * l, 1000 * w, 1000 * h,
          l / max(w, 1e-4), h / max(min(l, w), 1e-4),
          1.0 if obj.support_z - tz > 0.01 else 0.0,
          1.0 if any(x in name for x in ROUND_WORDS) else 0.0,
          1.0 if is_soft(name) else 0.0]


class BcGrasp:
  """JSON-weights MLP: features (8) -> [dx_mm, dy_mm, h_rel_top_mm, sin(2*jaw), cos(2*jaw), 3 force logits].

  The jaw angle has a 180-degree period (the jaws are symmetric), so it is regressed as
  (sin 2a, cos 2a) and decoded with atan2 - a plain MSE on degrees would break at the +-90 wrap."""

  def __init__(self, spec: dict):
    self.mean = np.asarray(spec["norm"]["mean"], dtype=np.float64)
    self.std = np.asarray(spec["norm"]["std"], dtype=np.float64)
    self.layers = [(np.asarray(L["W"], dtype=np.float64), np.asarray(L["b"], dtype=np.float64)) for L in spec["layers"]]
    self.meta = spec.get("meta", {})

  @classmethod
  def load(cls, path: Path | str = DEFAULT_MODEL) -> "BcGrasp | None":
    p = Path(path)
    if not p.exists():
      return None
    try:
      return cls(json.loads(p.read_text()))
    except (json.JSONDecodeError, KeyError, OSError):
      return None  # a broken model file must never break the pipeline; the ladder falls through

  def forward(self, feat) -> np.ndarray:
    x = (np.asarray(feat, dtype=np.float64) - self.mean) / self.std
    for i, (W, b) in enumerate(self.layers):
      x = x @ W + b
      if i < len(self.layers) - 1:
        x = np.maximum(x, 0.0)  # ReLU
    return x

  def predict(self, obj, tz: float) -> dict:
    """Grasp parameters for one measured object, in the `distill_grasp` format."""
    y = self.forward(features_for(obj, tz))
    jaw = float(np.degrees(np.arctan2(y[3], y[4])) / 2.0)
    return {"dxy_obj_mm": [round(float(y[0]), 1), round(float(y[1]), 1)],
            "h_rel_top_mm": round(float(y[2]), 1),
            "jaw_rel_deg": round((jaw + 90.0) % 180.0 - 90.0, 1),
            "force": FORCE_CLASSES[int(np.argmax(y[5:8]))]}
