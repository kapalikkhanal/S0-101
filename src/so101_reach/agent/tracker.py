"""Kalman-filter object tracking for the agent's perception.

The agent's perception measures each frame from scratch; `RealPerception` only bridges gaps
with a 3-frame zero-order hold. On the real arm the pipeline therefore parks the arm out of the
head camera's view before every measurement (~5-8 s each). `TrackedPerception` wraps any
perception (Real, Sim, NoisySim) with one Kalman filter per object track:

  - position: constant-velocity model, state [x y z vx vy vz], tuned for a near-static tabletop
    (small process noise, velocity damped). Covariance grows while an object is not measured,
    so confidence decays honestly during occlusion instead of a fixed frame count.
  - innovation gating: a measurement whose Mahalanobis distance exceeds the chi-square gate is
    rejected as an outlier (a wrong DINO box). Several consecutive rejections that agree with
    each other mean the object really moved (knocked over, slid): the track re-initialises.
  - yaw: scalar filter on the 180-degree-periodic footprint angle (wrapped innovation, own gate:
    near-square footprints flip by 90 degrees and that must not reject the position update).
  - size: scalar filters (the measured footprint changes slowly with viewing angle).

`find()` returns the filtered estimate as a `Detection` (extra["cov_mm"] = 1-sigma position
uncertainty). It sets extra["note"] ONLY when the estimate is stale or unconfident, because the
pipeline and the LLM read any note as "not measured now". `confident(names)` tells the pipeline
whether it may skip parking the arm; it is False for any object the arm came within
`touch_margin` of (measured from the object's half-diagonal), or that the pipeline `invalidate()`d after interacting with it (a failed grasp
can nudge an object without the camera seeing it). `forget(name)` drops a name's tracks (an
object in the gripper moves with it; a stale track would fake a placement check).
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field

import numpy as np

from so101_reach.agent.perception import Detection
from so101_reach.agent.sim_noise import occluded

CHI2_3_GATE = 16.27  # chi-square, 3 dof, p = 0.999
CHI2_1_GATE = 10.83  # chi-square, 1 dof, p = 0.999


def wrap90(a: float) -> float:
  return float((a + 90.0) % 180.0 - 90.0)


@dataclass
class KFCfg:
  meas_xy: float = 0.005  # measurement 1-sigma [m]
  meas_z: float = 0.003
  # Objects sit still unless the arm touches them, so the motion model is nearly a random walk
  # (~6 mm 1-sigma after a minute unseen). The ways an object really moves are handled
  # explicitly instead: arm contact (touch_radius / invalidate) and a seen jump (gate + re-init).
  accel: float = 0.0001  # white-noise acceleration [m/s^2]
  vel_damping_s: float = 2.0  # velocity decays with this time constant (no sustained drift)
  init_vel: float = 0.0001  # initial velocity 1-sigma [m/s]
  pos_walk: float = 0.0008  # position random walk [m/sqrt(s)] (settling, calibration drift)
  meas_yaw_deg: float = 4.0
  yaw_walk_deg: float = 1.0  # [deg/sqrt(s)]
  meas_size: float = 0.002
  size_walk: float = 0.0005  # [m/sqrt(s)]
  gate: float = CHI2_3_GATE
  reinit_after: int = 3  # consecutive mutually consistent rejections -> re-initialise
  confident_mm: float = 8.0  # max 1-sigma position error to skip a re-measurement
  stale_mm: float = 15.0  # above this the estimate is reported with a note
  max_age_s: float = 60.0  # never trust an estimate not measured for this long
  min_updates: int = 2  # a track is confirmed (reported) after this many measurements
  visible_misses: int = 3  # in clear view yet undetected this many frames -> track dropped
  occl_radius: float = 0.04
  touch_margin: float = 0.02  # arm this close to an object's surface: it may have nudged it -> re-measure first


class ObjectKF:
  """Per-object filter: constant-velocity position + yaw + size."""

  def __init__(self, det: Detection, t: float, cfg: KFCfg):
    self.cfg = cfg
    self.x = np.concatenate([np.asarray(det.center, float), np.zeros(3)])
    self.P = np.diag([cfg.meas_xy ** 2, cfg.meas_xy ** 2, cfg.meas_z ** 2] + [cfg.init_vel ** 2] * 3)
    self.R = np.diag([cfg.meas_xy ** 2, cfg.meas_xy ** 2, cfg.meas_z ** 2])
    self.yaw, self.p_yaw = float(det.yaw_deg), cfg.meas_yaw_deg ** 2
    self.size, self.p_size = np.asarray(det.size, float), np.full(3, cfg.meas_size ** 2)
    self.t = self.t_meas = t
    self.n_updates, self.rejects, self.misses = 1, [], 0
    self.touched = False  # the arm interacted with it since the last measurement
    self.last = det

  def predict(self, t: float) -> None:
    dt = max(0.0, t - self.t)
    if dt == 0.0:
      return
    c = self.cfg
    a = float(np.exp(-dt / c.vel_damping_s))
    F = np.eye(6)
    F[:3, 3:] = np.eye(3) * dt
    F[3:, 3:] = np.eye(3) * a
    q = c.accel ** 2
    Q = np.zeros((6, 6))
    Q[:3, :3] = np.eye(3) * (q * dt ** 3 / 3 + c.pos_walk ** 2 * dt)
    Q[:3, 3:] = Q[3:, :3] = np.eye(3) * q * dt ** 2 / 2
    Q[3:, 3:] = np.eye(3) * q * dt
    self.x = F @ self.x
    self.P = F @ self.P @ F.T + Q
    self.p_yaw += c.yaw_walk_deg ** 2 * dt
    self.p_size = self.p_size + c.size_walk ** 2 * dt
    self.t = t

  def mahalanobis2(self, z: np.ndarray) -> float:
    nu = z - self.x[:3]
    S = self.P[:3, :3] + self.R
    return float(nu @ np.linalg.solve(S, nu))

  def update(self, det: Detection, t: float) -> bool:
    """Predict to t and fuse a measurement; False if gated out as an outlier."""
    self.predict(t)
    z = np.asarray(det.center, float)
    if self.mahalanobis2(z) > self.cfg.gate:
      self.rejects.append(z)
      recent = self.rejects[-self.cfg.reinit_after:]
      if len(recent) >= self.cfg.reinit_after and max(np.linalg.norm(r - recent[-1]) for r in recent) < 0.02:
        self.__init__(det, t, self.cfg)  # the object really moved: start over from the new evidence
        return True
      return False
    self.rejects.clear()
    S = self.P[:3, :3] + self.R
    K = self.P[:, :3] @ np.linalg.inv(S)
    self.x = self.x + K @ (z - self.x[:3])
    I_KH = np.eye(6)
    I_KH[:, :3] -= K
    self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T  # Joseph form: stays symmetric PSD
    r_yaw = self.cfg.meas_yaw_deg ** 2
    nu = wrap90(float(det.yaw_deg) - self.yaw)
    if nu * nu / (self.p_yaw + r_yaw) <= CHI2_1_GATE:
      k = self.p_yaw / (self.p_yaw + r_yaw)
      self.yaw, self.p_yaw = wrap90(self.yaw + k * nu), (1 - k) * self.p_yaw
    r_s = self.cfg.meas_size ** 2
    k = self.p_size / (self.p_size + r_s)
    self.size = self.size + k * (np.asarray(det.size, float) - self.size)
    self.p_size = (1 - k) * self.p_size
    self.t_meas, self.n_updates, self.misses, self.touched = t, self.n_updates + 1, 0, False
    self.last = det
    return True

  def pos_sigma(self) -> float:
    """Largest 1-sigma position uncertainty [m]."""
    return float(np.sqrt(max(np.linalg.eigvalsh(self.P[:3, :3]).max(), 0.0)))

  def confirmed(self) -> bool:
    return self.n_updates >= self.cfg.min_updates

  def confident(self, t: float) -> bool:
    return (self.confirmed() and not self.touched and t - self.t_meas <= self.cfg.max_age_s
            and self.pos_sigma() * 1000 <= self.cfg.confident_mm)

  def detection(self, t: float, measured_now: bool) -> Detection:
    l, w, h = (float(v) for v in self.size)
    c = self.x[:3].copy()
    sig_mm = 1000 * self.pos_sigma()
    extra = {k: v for k, v in self.last.extra.items() if k not in ("note", "corners", "sim_noise")}
    extra["cov_mm"] = round(sig_mm, 1)
    extra["tracked"] = True
    age = t - self.t_meas
    if not measured_now and (sig_mm > self.cfg.stale_mm or age > self.cfg.max_age_s or not self.confirmed()):
      extra["note"] = f"not measured for {age:.0f} s - tracked estimate, +-{sig_mm:.0f} mm"
    return Detection(0, self.last.name, self.last.score, c, float(c[2] + h / 2), float(c[2] - h / 2), (l, w, h),
                     self.yaw, self.last.camera, self.last.bbox, extra)


@dataclass
class _NameTracks:
  tracks: list[ObjectKF] = field(default_factory=list)
  frame: int = -1  # last frame these tracks were updated on
  measured: set = field(default_factory=set)  # ids of tracks updated on that frame


class TrackedPerception:
  """Drop-in wrapper: same find / views / table_z (other attributes pass through to `per`)."""

  def __init__(self, per, names: list[str] | None = None, cfg: KFCfg | None = None, clock=None, arm_points=None):
    self.per = per
    self.cfg = cfg or KFCfg()
    self.names: dict[str, _NameTracks] = {n: _NameTracks() for n in (names or [])}
    self.frame = 0
    self._clock = clock
    self._arm_points = arm_points
    try:
      self._raw_kwargs = {"remember": False} if "remember" in inspect.signature(per.find).parameters else {}
    except (TypeError, ValueError):
      self._raw_kwargs = {}
    self.stats = {"updates": 0, "rejected": 0, "reinit": 0, "predicted_only": 0}

  def __getattr__(self, item):
    per = self.__dict__.get("per")
    if per is None:
      raise AttributeError(item)
    return getattr(per, item)

  # --- time / geometry --------------------------------------------------------------------
  def now(self) -> float:
    if self._clock is not None:
      return float(self._clock())
    world = getattr(self.per, "world", None)
    if world is not None and hasattr(world, "data"):
      return float(world.data.time)  # simulated seconds, not wall clock
    return time.monotonic()  # one clock for both fusion and confidence checks (camera stamps use a device clock)

  def _cam_t(self) -> np.ndarray | None:
    if hasattr(self.per, "cam_t"):
      return self.per.cam_t()
    world = getattr(self.per, "world", None)
    if world is not None:
      return world.camera_model("head")[1]
    cams = getattr(self.per, "cams", None)
    if cams and "head" in cams:
      return np.asarray(cams["head"][1].pose()[1], float)
    return None

  def _arm(self):
    f = self._arm_points or getattr(self.per, "arm_points", None)
    return f() if callable(f) else None

  def _hidden(self, p: np.ndarray, cam, arm) -> bool:
    return cam is not None and occluded(cam, p, arm, self.cfg.occl_radius)

  # --- tracking ---------------------------------------------------------------------------
  def _refresh(self, name: str) -> None:
    """Fuse this frame's raw detections of `name` into its tracks (once per frame)."""
    nt = self.names.setdefault(name, _NameTracks())
    if nt.frame == self.frame:
      return
    t = self.now()
    raw = self.per.find(name, **self._raw_kwargs)
    raw = [d for d in raw if not d.extra.get("note")]  # never fuse a memory/zero-order-hold result
    for tr in nt.tracks:
      tr.predict(t)
    free = list(range(len(nt.tracks)))
    nt.measured = set()
    for d in sorted(raw, key=lambda d: -d.score):
      z = np.asarray(d.center, float)
      best = min(free, key=lambda i: nt.tracks[i].mahalanobis2(z), default=None)
      if best is not None and nt.tracks[best].mahalanobis2(z) <= self.cfg.gate:
        nt.tracks[best].update(d, t)
        self.stats["updates"] += 1
        free.remove(best)
        nt.measured.add(id(nt.tracks[best]))
        continue
      # Gated out of every free track: an outlier of an existing track, or a new object.
      near = min(free, key=lambda i: float(np.linalg.norm(nt.tracks[i].x[:2] - z[:2])), default=None)
      if near is not None and nt.tracks[near].confirmed():
        tr = nt.tracks[near]
        if tr.update(d, t):  # re-initialised: consistent evidence it moved
          self.stats["reinit"] += 1
          nt.measured.add(id(tr))
          free.remove(near)
        else:
          self.stats["rejected"] += 1
        continue
      tr = ObjectKF(d, t, self.cfg)
      nt.tracks.append(tr)
      nt.measured.add(id(tr))
    cam, arm = self._cam_t(), self._arm()
    keep = []
    for tr in nt.tracks:
      if id(tr) not in nt.measured:
        self.stats["predicted_only"] += 1
        if not self._hidden(tr.x[:3], cam, arm):
          tr.misses += 1
      too_old = t - tr.t_meas > self.cfg.max_age_s
      gone = tr.misses >= self.cfg.visible_misses
      # A tentative (once-seen) track survives occlusion but not a miss in clear view.
      if not (too_old or gone) and (tr.confirmed() or id(tr) in nt.measured or tr.misses == 0):
        keep.append(tr)
    nt.tracks = keep
    nt.frame = self.frame

  def views(self):
    out = self.per.views()
    self.frame += 1
    return out

  def find(self, name: str, **_ignored) -> list[Detection]:
    self._refresh(name)
    nt = self.names[name]
    t = self.now()
    tracks = [tr for tr in nt.tracks if tr.confirmed()]
    if not tracks:  # nothing confirmed yet: tentative tracks (noted unless measured this frame)
      tracks = list(nt.tracks)
    # Measured this frame first, then the most recently measured, then detector score.
    tracks.sort(key=lambda tr: (id(tr) not in nt.measured, -tr.t_meas, -tr.last.score))
    return [tr.detection(t, id(tr) in nt.measured) for tr in tracks]

  def confident(self, names: list[str]) -> bool:
    """True if every name has a confirmed track whose estimate is good enough to act on
    without parking the arm and re-measuring."""
    t = self.now()
    for n in names:
      nt = self.names.get(n)
      if nt is None:
        return False
      self._mark_touched(nt)
      for tr in nt.tracks:
        tr.predict(t)  # uncertainty grown to now, not to the last frame
      if not any(tr.confident(t) for tr in nt.tracks):
        return False
    return True

  def _mark_touched(self, nt: _NameTracks) -> None:
    arm = self._arm()
    if arm is None or not len(arm):
      return
    for tr in nt.tracks:
      reach = float(np.linalg.norm(tr.size)) / 2 + self.cfg.touch_margin  # half-diagonal + margin
      if float(np.min(np.linalg.norm(np.asarray(arm) - tr.x[:3], axis=1))) < reach:
        tr.touched = True

  def invalidate(self, name: str) -> None:
    """The arm interacted with `name` (a grasp attempt, a release onto it): keep its tracks
    but do not trust them until the next measurement."""
    for tr in self.names.get(name, _NameTracks()).tracks:
      tr.touched = True

  def forget(self, name: str) -> None:
    if name in self.names:
      self.names[name] = _NameTracks()
