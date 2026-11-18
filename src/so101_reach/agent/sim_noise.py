"""Realistic measurement noise for the (otherwise exact) simulated perception.

`SimPerception` returns ground truth, so a tracker evaluated on it has nothing to do.
`NoisyPerception` wraps it with what the real head camera + Grounding DINO pipeline shows:

  - Gaussian noise on the centre (default sigma 5 mm in x/y, 3 mm in z), size (2 mm) and yaw (4 deg)
  - outliers: with probability `p_outlier` a measurement jumps 4-12 cm (a wrong DINO box)
  - occlusion dropout: no measurement while the arm passes between the head camera and the
    object (any arm point within `occl_radius` of the camera-object line) - the reason the
    pipeline parks the arm out of view before measuring on the real robot
  - optional random dropout `p_miss` (DINO missing an object in a frame)

Static surfaces (platforms, boxes) get the same noise but are never occluded-dropped, since the
real pipeline measures them large and reliably. Seeded, so evaluation runs are reproducible.
`log` records every find() result against ground truth for scripts/eval_tracker.py.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from so101_reach.agent.perception import Detection


@dataclass
class NoiseCfg:
  sigma_xy: float = 0.005
  sigma_z: float = 0.003
  sigma_size: float = 0.002
  sigma_yaw_deg: float = 4.0
  p_outlier: float = 0.03
  outlier_m: tuple[float, float] = (0.04, 0.12)
  p_miss: float = 0.02
  occl_radius: float = 0.035


def segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
  """Distance of each point (N, 3) to the segment a-b."""
  ab = b - a
  t = np.clip((points - a) @ ab / max(float(ab @ ab), 1e-12), 0.0, 1.0)
  return np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


def occluded(cam_t: np.ndarray, target: np.ndarray, arm_pts: np.ndarray | None, radius: float,
             margin: float = 0.03) -> bool:
  """True if an arm point lies within `radius` of the line of sight camera -> target. The last
  `margin` metres before the target are ignored (a gripper just above an object hides its top
  only partly; RealPerception drops detections the arm covers by more than a third)."""
  if arm_pts is None or len(arm_pts) == 0:
    return False
  d = target - cam_t
  n = float(np.linalg.norm(d))
  if n <= margin:
    return False
  end = cam_t + d * (1.0 - margin / n)
  return bool(np.min(segment_distance(np.asarray(arm_pts), cam_t, end)) < radius)


class NoisyPerception:
  """Wraps a SimPerception (same API: find, views, table_z, world)."""

  def __init__(self, per, arm_points=None, cfg: NoiseCfg | None = None, seed: int = 0, render: bool = True):
    self.per = per
    self.render = render  # False: views() returns no images (measurement-only evaluation, much faster)
    self.world = per.world
    self.table_z = per.table_z
    self.arm_points = arm_points  # () -> (N, 3) or None; set after the arm exists
    self.cfg = cfg or NoiseCfg()
    self.rng = np.random.default_rng(seed)
    self.log: list[dict] = []  # one entry per object find(): truth vs what was returned

  def cam_t(self) -> np.ndarray:
    return self.world.camera_model("head")[1]

  def views(self):
    return self.per.views() if self.render else {}

  def _noisy(self, d: Detection) -> Detection:
    c, r = self.cfg, self.rng
    if r.uniform() < c.p_outlier:
      ang = r.uniform(0, 2 * np.pi)
      jump = r.uniform(*c.outlier_m)
      dc = np.array([jump * np.cos(ang), jump * np.sin(ang), r.normal(0, 0.01)])
      kind = "outlier"
    else:
      dc = r.normal(0, [c.sigma_xy, c.sigma_xy, c.sigma_z])
      kind = "noise"
    center = np.asarray(d.center, float) + dc
    size = tuple(max(0.004, float(s + r.normal(0, c.sigma_size))) for s in d.size)
    yaw = float((d.yaw_deg + r.normal(0, c.sigma_yaw_deg) + 90) % 180 - 90)
    half_h = size[2] / 2
    extra = dict(d.extra)
    extra.pop("corners", None)  # exact corners would leak ground truth into annotations
    extra["sim_noise"] = kind
    return replace(d, center=center, top_z=float(center[2] + half_h), support_z=float(center[2] - half_h),
                   size=size, yaw_deg=yaw, extra=extra)

  def find(self, name: str, **_ignored) -> list[Detection]:
    labels = set(self.world.labels)
    arm = self.arm_points() if self.arm_points is not None else None
    cam = self.cam_t()
    out = []
    for d in self.per.find(name):
      if d.name not in labels:  # static surface
        out.append(self._noisy(d))
        continue
      truth = np.asarray(d.center, float)
      hidden = occluded(cam, truth, arm, self.cfg.occl_radius)
      missed = hidden or self.rng.uniform() < self.cfg.p_miss
      nd = None if missed else self._noisy(d)
      self.log.append({"t": float(self.world.data.time), "query": name, "label": d.name, "truth": truth.tolist(),
                       "truth_yaw": float(d.yaw_deg), "occluded": hidden,
                       "meas": None if nd is None else nd.center.tolist(),
                       "kind": "occluded" if hidden else ("miss" if nd is None else nd.extra["sim_noise"])})
      if nd is not None:
        out.append(nd)
    return out
