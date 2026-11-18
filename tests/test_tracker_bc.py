"""Unit checks for the Kalman tracker, the sim noise geometry and the BC grasp model.

numpy-only (no MuJoCo, no GPU):  PYTHONPATH=src python3 -m pytest tests/test_tracker_bc.py -q
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from so101_reach.agent.bc_grasp import FEATURE_NAMES, FORCE_CLASSES, BcGrasp, features_for
from so101_reach.agent.perception import Detection
from so101_reach.agent.sim_noise import occluded
from so101_reach.agent.tracker import KFCfg, ObjectKF, TrackedPerception, wrap90

ROOT = Path(__file__).resolve().parents[1]


def det(c, yaw=10.0, size=(0.034, 0.034, 0.034), name="red cube", score=0.9, note=None):
  extra = {"note": note} if note else {}
  c = np.asarray(c, float)
  return Detection(0, name, score, c, float(c[2] + size[2] / 2), float(c[2] - size[2] / 2), size, yaw, "head", None, extra)


class FakePer:
  """Scripted perception: `script[frame]` is the list of detections find() returns."""

  def __init__(self):
    self.table_z, self.t, self.frame = -0.0132, 0.0, 0
    self.script: dict[int, list[Detection]] = {}

  def views(self):
    self.frame += 1
    return {}

  def find(self, name):
    return list(self.script.get(self.frame, []))


def make(per, **kw):
  return TrackedPerception(per, ["red cube"], cfg=KFCfg(**kw), clock=lambda: per.t)


# --- Kalman filter -----------------------------------------------------------------------
def test_converges_on_noisy_static_object():
  rng = np.random.default_rng(0)
  truth = np.array([0.25, -0.05, 0.004])
  kf = ObjectKF(det(truth + rng.normal(0, 0.005, 3)), 0.0, KFCfg())
  raw = []
  for k in range(1, 40):
    z = truth + rng.normal(0, [0.005, 0.005, 0.003])
    raw.append(np.linalg.norm(z - truth))
    assert kf.update(det(z), 0.5 * k)
  assert np.linalg.norm(kf.x[:3] - truth) < 0.5 * np.mean(raw)
  assert kf.pos_sigma() < 0.003


def test_gate_rejects_outlier_and_reinitialises_on_consistent_jump():
  kf = ObjectKF(det([0.25, 0.0, 0.0]), 0.0, KFCfg())
  for k in range(1, 10):
    kf.update(det([0.25, 0.0, 0.0]), 0.5 * k)
  before = kf.x[:3].copy()
  assert not kf.update(det([0.33, 0.0, 0.0]), 5.0)  # 8 cm jump: rejected
  assert np.allclose(kf.x[:3], before, atol=1e-3)
  assert not kf.update(det([0.33, 0.0, 0.0]), 5.5)
  assert kf.update(det([0.33, 0.0, 0.0]), 6.0)  # third consistent rejection: it really moved
  assert np.allclose(kf.x[:3], [0.33, 0.0, 0.0], atol=1e-6)
  assert kf.n_updates == 1


def test_covariance_grows_without_updates():
  kf = ObjectKF(det([0.2, 0.0, 0.0]), 0.0, KFCfg())
  for k in range(1, 6):
    kf.update(det([0.2, 0.0, 0.0]), float(k))
  s0 = kf.pos_sigma()
  kf.predict(25.0)
  s1 = kf.pos_sigma()
  kf.predict(60.0)
  assert s0 < s1 < kf.pos_sigma()


def test_yaw_wraps_at_90():
  kf = ObjectKF(det([0.2, 0.0, 0.0], yaw=88.0), 0.0, KFCfg())
  for k in range(1, 20):
    kf.update(det([0.2, 0.0, 0.0], yaw=-88.0 if k % 2 else 88.0), float(k))
  assert abs(wrap90(kf.yaw - 90.0)) < 3.0  # stays near +-90, never averages to 0
  assert wrap90(179.0) == -1.0 and wrap90(-91.0) == 89.0


# --- TrackedPerception -------------------------------------------------------------------
def test_tracked_perception_bridges_occlusion_and_confidence():
  per = FakePer()
  tp = make(per)
  for k in range(1, 6):
    per.t = float(k)
    per.script[k] = [det([0.25, -0.05, 0.004])]
    tp.views()
    out = tp.find("red cube")
    assert len(out) == 1 and "note" not in out[0].extra
  assert tp.confident(["red cube"])
  assert not tp.confident(["red cube", "white box"])
  per.t = 7.0
  tp.views()  # no detection this frame (arm in the way)
  out = tp.find("red cube")
  assert len(out) == 1 and "note" not in out[0].extra  # still confident: no note
  assert np.allclose(out[0].center, [0.25, -0.05, 0.004], atol=0.003)
  per.t = 200.0
  assert not tp.confident(["red cube"])  # far too old


def test_touch_and_invalidate_require_a_fresh_measurement():
  per = FakePer()
  arm = {"pts": np.array([[0.0, 0.0, 0.3]])}
  tp = TrackedPerception(per, ["red cube"], clock=lambda: per.t, arm_points=lambda: arm["pts"])
  for k in range(1, 4):
    per.t, per.script[k] = float(k), [det([0.25, 0.0, 0.0])]
    tp.views()
    tp.find("red cube")
  assert tp.confident(["red cube"])
  tp.invalidate("red cube")
  assert not tp.confident(["red cube"])
  per.t, per.script[4] = 4.0, [det([0.25, 0.0, 0.0])]
  tp.views()
  tp.find("red cube")
  assert tp.confident(["red cube"])
  arm["pts"] = np.array([[0.25, 0.0, 0.04]])  # gripper right above it
  assert not tp.confident(["red cube"])


def test_outlier_does_not_become_the_reported_detection():
  per = FakePer()
  tp = make(per)
  for k in range(1, 6):
    per.t = float(k)
    per.script[k] = [det([0.25, 0.0, 0.0])]
    tp.views()
    tp.find("red cube")
  per.t, per.script[6] = 6.0, [det([0.10, 0.10, 0.0], score=0.99)]
  tp.views()
  out = tp.find("red cube")
  assert np.allclose(out[0].center[:2], [0.25, 0.0], atol=0.003)
  assert tp.stats["rejected"] == 1


def test_forget_and_memory_results_are_not_fused():
  per = FakePer()
  tp = make(per)
  per.t, per.script[1] = 1.0, [det([0.2, 0.0, 0.0])]
  tp.views()
  tp.find("red cube")
  tp.forget("red cube")
  per.t, per.script[2] = 2.0, [det([0.2, 0.0, 0.0], note="not visible now (hidden by the arm?)")]
  tp.views()
  assert tp.find("red cube") == []


def test_tentative_track_has_note_when_not_measured():
  per = FakePer()
  tp = make(per)
  per.t, per.script[1] = 1.0, [det([0.2, 0.0, 0.0])]
  tp.views()
  assert "note" not in tp.find("red cube")[0].extra  # fresh measurement
  per.t = 2.0
  tp.views()
  assert tp.find("red cube") == []  # one sighting, then gone in clear view: not confirmed


# --- sim noise geometry ------------------------------------------------------------------
def test_occlusion_line_of_sight():
  cam = np.array([0.0, 0.0, 0.5])
  obj = np.array([0.3, 0.0, 0.0])
  mid = (cam + obj) / 2
  assert occluded(cam, obj, np.array([mid]), 0.03)
  assert not occluded(cam, obj, np.array([mid + [0, 0.1, 0]]), 0.03)
  assert not occluded(cam, obj, np.array([obj + 0.2 * (obj - cam)]), 0.03)  # behind the object
  assert not occluded(cam, obj, np.array([obj + [0.0, 0.035, 0.0]]), 0.03)  # beside it, inside the end margin
  assert not occluded(cam, obj, None, 0.03)


class _FakeWorld:
  """Just enough of SimWorld for SimPerception + NoisyPerception."""

  def __init__(self):
    self.labels = ["red cube"]
    self.cfg = type("C", (), {"table_z": -0.0132})()
    self.data = type("D", (), {"time": 0.0})()
    self.c = np.array([0.25, -0.05, 0.004])

  def object_truth(self, label):
    return {"center": self.c.copy(), "top": 0.021, "bottom": -0.013, "size": [0.034] * 3, "yaw_deg": 20.0,
            "corners": np.zeros((8, 3))}

  def surfaces(self):
    return []

  def camera_model(self, cam):
    return np.eye(3), np.array([0.0, 0.0, 0.5]), 500.0, 500.0, 320.0, 240.0


def test_noisy_sim_into_tracker_wiring():
  from so101_reach.agent.perception import SimPerception
  from so101_reach.agent.sim_noise import NoiseCfg, NoisyPerception

  w = _FakeWorld()
  arm = {"pts": np.array([[0.0, 0.3, 0.2]])}
  noisy = NoisyPerception(SimPerception(w), arm_points=lambda: arm["pts"], cfg=NoiseCfg(p_outlier=0.0, p_miss=0.0),
                          seed=0, render=False)
  tp = TrackedPerception(noisy, ["red cube"], arm_points=lambda: arm["pts"])
  errs = []
  for k in range(30):
    w.data.time = 0.5 * k
    tp.views()
    d = tp.find("red cube")[0]
    errs.append(np.linalg.norm(d.center[:2] - w.c[:2]))
  assert tp.now() == w.data.time  # simulated clock, not wall time
  assert np.mean(errs[-10:]) < 0.004 and tp.confident(["red cube"])
  arm["pts"] = np.array([(np.array([0.0, 0.0, 0.5]) + w.c) / 2])  # arm on the line of sight
  w.data.time += 0.5
  tp.views()
  d = tp.find("red cube")[0]
  assert noisy.log[-1]["kind"] == "occluded"
  assert "note" not in d.extra and np.linalg.norm(d.center[:2] - w.c[:2]) < 0.006
  assert tp.stats["predicted_only"] >= 1


# --- BC model ----------------------------------------------------------------------------
def _load_trainer():
  spec = importlib.util.spec_from_file_location("train_bc_grasp", ROOT / "scripts" / "train_bc_grasp.py")
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def test_bc_train_save_load_roundtrip(tmp_path):
  T = _load_trainer()
  rng = np.random.default_rng(0)
  n = 300
  X = np.column_stack([rng.uniform(24, 44, n), rng.uniform(10, 44, n), rng.uniform(20, 44, n),
                       rng.uniform(1, 3, n), rng.uniform(0.5, 2, n), rng.integers(0, 2, n),
                       rng.integers(0, 2, n), np.zeros(n)])
  jaw = np.clip(0.5 * (X[:, 0] - 34), -40, 40)
  Y = np.column_stack([0.1 * (X[:, 0] - 34), -0.05 * X[:, 1] + 1, -0.4 * X[:, 2] + 5,
                       np.sin(np.radians(2 * jaw)), np.cos(np.radians(2 * jaw))])
  F = (X[:, 2] > 32).astype(int)
  layers, norm = T.train(X[:240], Y[:240], F[:240], epochs=300, seed=0)
  path = tmp_path / "bc.json"
  path.write_text(json.dumps(T.to_spec(layers, norm, {"note": "unit"})))
  model = BcGrasp.load(path)
  assert model is not None and model.meta["note"] == "unit"
  m = T.metrics(model, X[240:], Y[240:], F[240:])
  base = T.mean_baseline(Y[:240], F[:240], X[240:], Y[240:], F[240:])
  assert m["h_mae_mm"] < 0.5 * base["h_mae_mm"]
  assert m["force_acc"] > 0.85
  # numpy forward in bc_grasp == the trainer's forward (same layers, same normalization)
  xn = (X[240:] - np.array(norm["mean"])) / np.array(norm["std"])
  ref, _ = T.forward([[np.asarray(W), np.asarray(b)] for W, b in layers], xn)
  ours = np.array([model.forward(x) for x in X[240:]])
  assert np.allclose(ref, ours, atol=1e-9)


def test_bc_predict_format_and_features():
  obj = det([0.25, 0.0, 0.004], size=(0.05, 0.02, 0.03), name="red bar")
  f = features_for(obj, -0.0132)
  assert len(f) == len(FEATURE_NAMES)
  rng = np.random.default_rng(1)
  spec = {"norm": {"mean": [0.0] * 8, "std": [1.0] * 8},
          "layers": [{"W": rng.normal(size=(8, 4)).tolist(), "b": [0.0] * 4},
                     {"W": rng.normal(size=(4, 8)).tolist(), "b": [0.0] * 8}]}
  p = BcGrasp(spec).predict(obj, -0.0132)
  assert set(p) == {"dxy_obj_mm", "h_rel_top_mm", "jaw_rel_deg", "force"}
  assert -90 <= p["jaw_rel_deg"] < 90 and p["force"] in FORCE_CLASSES and len(p["dxy_obj_mm"]) == 2


def test_bc_load_missing_or_broken(tmp_path):
  assert BcGrasp.load(tmp_path / "nope.json") is None
  (tmp_path / "bad.json").write_text("{not json")
  assert BcGrasp.load(tmp_path / "bad.json") is None


def test_jaw_fold_symmetry():
  from so101_reach.agent.local_skills import fold_jaw_rel, square_footprint
  assert square_footprint(0.03, 0.03) and square_footprint(0.03, 0.035) and not square_footprint(0.06, 0.02)
  # square / round: 0 and 90 deg are the same grasp -> one label
  assert fold_jaw_rel(89.0, 0.03, 0.03) == fold_jaw_rel(-1.0, 0.03, 0.03) == -1.0
  assert fold_jaw_rel(-90.0, 0.03, 0.03) == 0.0
  # elongated: 180 deg period only (across the narrow side matters)
  assert fold_jaw_rel(89.0, 0.06, 0.02) == 89.0
  assert fold_jaw_rel(-91.0, 0.06, 0.02) == 89.0


def test_trainer_folds_square_jaw_labels():
  T = _load_trainer()
  feat_sq = [30.0, 30.0, 30.0, 1.0, 1.0, 0.0, 0.0, 0.0]
  rows = [{"success": True, "seed": i, "features": feat_sq,
           "params": {"dxy_obj_mm": [0, 0], "h_rel_top_mm": -15.0, "jaw_rel_deg": j, "force": "firm"}}
          for i, j in enumerate([0.0, 89.0, -90.0, 1.0])]
  _, Y, _, _ = T.to_arrays(rows)
  jaw = np.degrees(np.arctan2(Y[:, 3], Y[:, 4])) / 2
  assert np.all(np.abs(jaw) <= 1.0 + 1e-9)  # all four collapse onto ~0 deg
