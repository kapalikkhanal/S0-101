"""Timestamped scene snapshots. Camera processing never reads the servo bus.

Simulation uses explicitly labelled ground truth, for controller/skill evaluation.
The hardware source runs Grounding DINO periodically, optical flow between detections,
depth geometry on each tracked box and an unlabelled obstacle cloud in a worker.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import threading
import time

import cv2
import numpy as np

from so101_reach.agent.perception import Detection, SimPerception, measure_object


@dataclass
class Snapshot:
  stamp: float
  objects: dict[str, Detection]
  seen: dict[str, float]
  cloud: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
  image: np.ndarray | None = None
  frame: int = 0
  error: str | None = None
  occluded: dict[str, float] = field(default_factory=dict)  # name -> time it became hidden by the arm
  joints: np.ndarray | None = None  # measured arm pose associated with this depth frame
  hands: list = field(default_factory=list)  # detected hands: {"bbox", "score", "center" (m or None)}


class SceneUnavailable(RuntimeError):
  pass


def fresh_object(s: Snapshot, name: str, now: float, max_age: float = 0.8, allow_occluded: float = 0.0) -> Detection:
  """The object's current measurement. allow_occluded > 0: if the arm itself hides the object,
  its last fresh measurement (at most allow_occluded seconds old) is returned, flagged with
  extra['occluded_s']. The scene itself must still be fresh; nothing else relaxes."""
  if s.error:
    raise SceneUnavailable(s.error)
  if now - s.stamp > max_age:
    raise SceneUnavailable(f"scene stale by {now - s.stamp:.2f}s")
  if name not in s.objects or now - s.seen.get(name, -float("inf")) > max_age:
    age = now - s.seen.get(name, -float("inf"))
    if allow_occluded > 0 and name in s.objects and name in s.occluded and age <= allow_occluded:
      d = s.objects[name]
      return replace(d, extra={**d.extra, "occluded_s": round(float(age), 3)})
    raise SceneUnavailable(f"{name!r} not freshly observed" + (" (hidden by the arm)" if name in s.occluded else ""))
  d = s.objects[name]
  if d.extra.get("note"):
    raise SceneUnavailable(f"{name!r} is remembered, not measured")
  return d


def footprint_contains(d: Detection, xy: np.ndarray, half_extent=0.0, margin=0.0) -> bool:
  a = np.radians(d.yaw_deg)
  rot = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
  p = np.abs(rot @ (np.asarray(xy) - d.center[:2]))
  return bool(np.all(p + half_extent <= np.array(d.size[:2]) / 2 - margin))


ROBOT_BODIES = ("base","shoulder","upper_arm","lower_arm","wrist","gripper","moving_jaw_so101_v1","camera_mount")


def attached_boxes():
  """Measured robot-attached hardware missing from the model (configs/attached_hardware.json)."""
  import json
  from so101_reach.sim import so101
  path = so101.PROJECT_ROOT / "configs" / "attached_hardware.json"
  if not path.exists():
    return []
  return [(b["body"], np.array(b["center_m"], float), np.array(b["half_m"], float), b["name"]) for b in json.loads(path.read_text())["boxes"]]


def _voxel(points: np.ndarray, size: float) -> np.ndarray:
  if not len(points):
    return points
  _, idx = np.unique(np.floor(points / size).astype(np.int64), axis=0, return_index=True)
  return points[idx]


class RobotSurface:
  """Points on the robot's own surfaces (visual meshes + collision primitives), for removing
  the arm from the observed obstacle cloud at the joint state of a camera frame."""

  def __init__(self, m, spacing: float = 0.004):
    import mujoco
    bodies = {m.body(n).id for n in ROBOT_BODIES}
    self.items = []
    for g in range(m.ngeom):
      if m.geom_bodyid[g] not in bodies:
        continue
      kind, size = int(m.geom_type[g]), m.geom_size[g]
      if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mid = m.geom_dataid[g]
        pts = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
      elif kind == mujoco.mjtGeom.mjGEOM_BOX:
        axes = [np.arange(-s, s + 1e-9, spacing) if s > spacing else np.array([0.0]) for s in size]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        pts = grid[np.any(np.isclose(np.abs(grid), size, atol=spacing / 2), axis=1)] if len(grid) > 1 else grid
      elif kind in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        r = size[0]; hl = 0.0 if kind == mujoco.mjtGeom.mjGEOM_SPHERE else size[1]
        n = max(8, int(4 * np.pi * r * r / spacing ** 2) + int(2 * np.pi * r * 2 * hl / spacing ** 2))
        k = np.arange(n) + 0.5
        phi, th = np.arccos(1 - 2 * k / n), np.pi * (1 + 5 ** 0.5) * k
        unit = np.column_stack((np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)))
        pts = r * unit
        if hl:
          pts = np.concatenate([pts + [0, 0, hl] * (unit[:, 2:] >= 0) - [0, 0, hl] * (unit[:, 2:] < 0),
                                np.column_stack((r * np.cos(th), r * np.sin(th), np.interp(k, [0, n], [-hl, hl])))])
      else:
        continue
      self.items.append((g, _voxel(np.asarray(pts, float), spacing)))
    distal = {m.body(n).id for n in ("gripper", "moving_jaw_so101_v1")}
    self.proximal_geom = {g for g, _ in self.items if m.geom_bodyid[g] not in distal}
    # Attached hardware: filled grids (points inside the volume must be removed too).
    self.attached = []
    for body, c, h, _ in attached_boxes():
      axes = [np.arange(-x, x + 1e-9, 0.006) for x in h]
      grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3) + c
      self.attached.append((m.body(body).id, grid))
    self.count = sum(len(p) for _, p in self.items) + sum(len(p) for _, p in self.attached)

  def world(self, d, proximal: bool | None = None) -> np.ndarray:
    """All surface points; proximal=True: base..wrist links only; False: gripper, jaws, attached."""
    parts = [d.geom_xpos[g] + pts @ d.geom_xmat[g].reshape(3, 3).T for g, pts in self.items
             if proximal is None or (g in self.proximal_geom) == proximal]
    if proximal is not True:
      parts += [d.xpos[b] + pts @ d.xmat[b].reshape(3, 3).T for b, pts in self.attached]
    return np.concatenate(parts)


def size_consistent(a, b, abs_tol: float = 0.010, rel_tol: float = 0.25) -> bool:
  """Rigid objects keep their size: footprint (sorted) and height within max(abs, rel) tolerance."""
  sa = np.array([*sorted(a[:2], reverse=True), a[2]]); sb = np.array([*sorted(b[:2], reverse=True), b[2]])
  return bool(np.all(np.abs(sa - sb) <= np.maximum(abs_tol, rel_tol * np.maximum(sa, sb))))


def same_object(a: Detection, b: Detection) -> bool:
  if a.bbox is not None and b.bbox is not None:
    x0,y0=np.maximum(a.bbox[:2],b.bbox[:2]);x1,y1=np.minimum(a.bbox[2:],b.bbox[2:])
    inter=max(0,x1-x0)*max(0,y1-y0)
    union=np.prod(a.bbox[2:]-a.bbox[:2])+np.prod(b.bbox[2:]-b.bbox[:2])-inter
    if union>0 and inter/union>0.6:
      return True
  return bool(np.linalg.norm(a.center-b.center)<0.015)


def assign_detections(candidates: dict[str, list[Detection]], margin: float = 0.10):
  """One-to-one assignment of requested names to physical detections.

  Different names in one instruction denote different objects, so detections are assigned
  jointly to maximise the total detector score. A name is accepted only if every assignment
  giving it a different object (or none) scores at least `margin` lower; otherwise it stays
  ambiguous and is not reported. Returns ({name: Detection}, {name: margin or reason})."""
  names=list(candidates)
  objs=[]  # physical identities across all names' candidate lists
  for n in names:
    for d in candidates[n]:
      if not any(same_object(d,o) for o in objs):objs.append(d)
  score={(n,i):max([d.score for d in candidates[n] if same_object(d,o)],default=None) for n in names for i,o in enumerate(objs)}
  options={n:[i for i in range(len(objs)) if score[(n,i)] is not None]+[None] for n in names}
  totals=[]
  def search(k,used,picked,total):
    if k==len(names):
      totals.append((total,dict(picked)));return
    n=names[k]
    for i in options[n]:
      if i is not None and i in used:continue
      picked[n]=i
      search(k+1,used|({i} if i is not None else set()),picked,total+(score[(n,i)] if i is not None else 0.0))
    picked.pop(n,None)
  if len(names)>6 or len(objs)>10:
    raise ValueError("too many names/candidates for exhaustive assignment")
  search(0,frozenset(),{},0.0)
  best_total,best=max(totals,key=lambda x:x[0])
  out,info={},{}
  for n in names:
    i=best[n]
    if i is None:
      info[n]="no detection";continue
    rival=max([t for t,a in totals if a[n]!=i],default=-np.inf)
    gap=best_total-rival
    if gap<margin:
      info[n]=f"ambiguous (assignment margin {gap:.2f})";continue
    d=max((d for d in candidates[n] if same_object(d,objs[i])),key=lambda d:d.score)
    out[n]=d;info[n]=round(float(gap),3)
  return out,info


class SimScene:
  def __init__(self, world, names, perturb="none"):
    self.world, self.names = world, names
    self.per = SimPerception(world)
    self.table_z = world.cfg.table_z
    self.perturb = perturb
    self.perturbed = False
    self.events = []

  def now(self):
    return float(self.world.data.time)

  def placement_point(self,target,payload,occupants=()):
    from so101_reach.manipulation.destination import packed_release_point
    occupants=list(occupants)
    for name in self.world.labels:
      d=self.per.find(name)
      if d:occupants.append(d[0])
    return packed_release_point(target,payload,occupants)

  def snapshot(self):
    objs = {}
    for n in self.names:
      ds = self.per.find(n)
      exact = [d for d in ds if d.name == n]
      if len(exact) == 1:
        objs[n] = exact[0]
    if self.perturb == "occlusion" and self.now() >= 2.5:
      objs.pop(self.names[0], None)
    return Snapshot(self.now(), objs, {n: self.now() for n in objs}, frame=int(self.now() / 0.02))

  def after_step(self):
    if self.perturbed or self.now() < 2.5 or self.perturb not in ("target", "obstacle"):
      return
    import mujoco
    w = self.world
    name = self.names[0] if self.perturb == "target" else "moving obstacle"
    if name not in w.obj_body:
      return
    bid = w.obj_body[name]
    adr = w.model.jnt_qposadr[w.model.body_jntadr[bid]]
    if self.perturb == "target":
      w.data.qpos[adr + 1] += 0.035
    else:
      w.data.qpos[adr:adr + 3] = (0.29, -0.08, self.table_z + 0.10)
    mujoco.mj_forward(w.model, w.data)
    self.perturbed = True
    self.events.append({"kind": "perturbation", "type": self.perturb, "sim_s": self.now()})

  def set_joints(self, q6):
    pass

  def close(self):
    pass


def partial_consistent(depth, v, box, old: Detection, tol: float = 0.006, min_points: int = 25, min_inside: float = 0.8,
                       robot=None, stats=None, min_occupied: float = 0.6) -> str:
  """'consistent' if visible (unmasked) object points in `box` lie inside `old`'s oriented box
  (inflated by tol); 'inconsistent' if enough points disagree; 'hidden' if too few are visible."""
  h, w = depth.shape
  x0, y0, x1, y1 = np.rint(box).astype(int)
  vs, us = np.mgrid[max(0, y0):min(h, y1):2, max(0, x0):min(w, x1):2]
  z = depth[vs, us]
  ok = z > 0
  if ok.sum() < min_points:
    return "hidden"
  pts = np.column_stack(((us[ok] - v.cx) / v.fx * z[ok], (vs[ok] - v.cy) / v.fy * z[ok], z[ok])) @ v.R.T + v.t
  pts = pts[pts[:, 2] > old.support_z + tol]  # object surface, not its support
  if robot is not None and len(pts):
    pts = pts[~robot(pts)]  # the arm's own cables/links are not evidence about the object
  # Points well above the object are occluders in front of it (an arm carrying a payload over the
  # box made the unmoved box "inconsistent"), not evidence about where it is.
  pts = pts[pts[:, 2] <= old.top_z + 0.02]
  if len(pts) < min_points:
    return "hidden"
  a = np.radians(old.yaw_deg)
  local = (pts[:, :2] - old.center[:2]) @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
  inside = np.all(np.abs(local) <= np.array(old.size[:2]) / 2 + tol, axis=1) & (pts[:, 2] <= old.top_z + tol)
  if stats is not None:
    stats.update(n=int(len(pts)), inside=round(float(inside.mean()), 2))
  if inside.mean() < min_inside:
    return "inconsistent"
  if old.extra.get("container"):
    return "consistent"  # an open container's footprint centre is its floor, not a top surface
  # Converse: where the old top surface is visible, it must still be there (a shift exposes support).
  # Central 60 % of the footprint only: real depth is unreliable within a few mm of edges
  # (measured 2026-11-07: edge samples read the table and failed an unmoved cube).
  axes = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
  half = np.maximum(0.6 * np.array(old.size[:2]) / 2, 0.002)
  grid = np.stack(np.meshgrid(np.linspace(-1, 1, 7), np.linspace(-1, 1, 7)), -1).reshape(-1, 2) * half
  vis = occ = 0
  for xy in old.center[:2] + grid @ axes:
    px = v.project(np.array([*xy, old.top_z]))
    if px is None or not (0 <= px[1] < h and 0 <= px[0] < w) or depth[px[1], px[0]] <= 0:
      continue
    zz = depth[px[1], px[0]]
    q = v.R @ np.array([(px[0] - v.cx) / v.fx * zz, (px[1] - v.cy) / v.fy * zz, zz]) + v.t
    vis += 1
    occ += q[2] > old.top_z - tol
  if stats is not None:
    stats.update(top_visible=vis, top_occupied=round(occ / vis, 2) if vis else None)
  # Unmoved cube on hardware: 0.65-0.88 occupied; a 2.5 cm shift empties the central samples (0.29).
  if vis >= 8 and occ / vis < min_occupied:
    return "inconsistent"
  return "consistent"


def flow_shift(prev_gray: np.ndarray, gray: np.ndarray, box: np.ndarray, max_shift: float = 60.0, exclude=None):
  """Median forward-backward-consistent LK shift of features inside `box` (outside the boolean
  `exclude` mask, e.g. the arm); None if unreliable."""
  mask = np.zeros_like(prev_gray)
  x0, y0, x1, y1 = np.rint(box).astype(int)
  mask[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 255
  if exclude is not None:
    mask[exclude] = 0
  pp = cv2.goodFeaturesToTrack(prev_gray, maxCorners=40, qualityLevel=0.02, minDistance=5, mask=mask)
  if pp is None or len(pp) < 4:
    return None
  qq, ok, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pp, None)
  back, bok, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, qq, None)
  valid = ok.ravel().astype(bool) & bok.ravel().astype(bool) & (np.linalg.norm(back - pp, axis=2).ravel() < 1.5)
  if valid.sum() < 4:
    return None
  shift = np.median((qq - pp)[valid, 0], axis=0)
  return None if np.linalg.norm(shift) > max_shift else shift


class CameraScene:
  """Head RGB-D worker. Two threads, neither touching the servo bus or main-thread MjData:

  * observation loop (every frame): optical-flow box tracking, depth re-measurement of each
    tracked box, arm masking, unlabelled obstacle cloud with a 3-D robot self-filter; publishes
    a timestamped Snapshot. An object is fresh only when its geometry was re-measured.
  * detector thread: Grounding DINO on a copy of a recent frame + one-to-one name assignment.
    Its boxes are flow-propagated from their source frame to the current frame before use, so
    detector latency delays identity updates but does not age the published geometry.
  """

  def __init__(self, perception, names: list[str], detection_period=0.6, assign_margin=0.10):
    self.per, self.names = perception, names
    self.assign_margin = assign_margin
    self.max_width = {}  # name -> largest measured footprint width a detection may have (pick targets)
    self.max_reach = {}  # name -> (shoulder axis xy, radius m): pick-target detections the arm can reach
    self.gate_rejections = {}  # name -> latest reason a detection was not a pick candidate
    self.assignment = {}
    self.self_pad = 0.015
    self.proximal_pad = 0.035
    self.self_filtered = 0
    self.rejected_identity = []
    self.watch_hands = True
    self.hand_threshold = 0.40
    self._hands = (-1e9, [])
    self.q_hist = []
    self.latency_window = 0.25
    import json
    from so101_reach.sim import so101
    ws = so101.PROJECT_ROOT / "configs" / "real_workspace.json"
    w = json.loads(ws.read_text()) if ws.exists() else None
    self.workspace = (np.array([w["x_min"], w["y_min"]]), np.array([w["x_max"], w["y_max"]])) if w else None
    self.identity_max_age = 10.0  # since the last measurement, partial verification or detection
    # A track the arm itself hides cannot be confirmed; its box is frozen. Keep its identity as
    # long as the executor may use a hidden pick target (WRIST_CONFIRMED_ALLOWANCE_S, 30 s; the
    # executor's own check still bounds use to 8 s without a wrist camera). With 10 s, a slow RL
    # approach (pick_place_fsm_real/rl_2) lost the cube while passing over it.
    self.occluded_identity_max_age = 30.0
    self.timeline = []  # (time, {name: measured|consistent|hidden|inconsistent|untracked|no_depth|no_identity}) on change
    self.loop_s = []
    self.timing: dict[str, list[float]] = {}
    self.table_z = perception.table_z
    self.period = detection_period
    self.lock = threading.Lock()
    self.stop = threading.Event()
    self._snapshot = Snapshot(0, {}, {})
    self.q6 = None
    self._points = None
    self._det_lock = threading.Lock()
    self._det_request = None   # (frame_no, Frame) for the detector
    self._det_result = None    # (frame_no, {name: Detection}, info)
    self._det_wake = threading.Event()
    self._forget = set()
    self._held = set()
    self._fixtures = {}  # verified destination geometry; partial views cannot rewrite its shape/type
    self._placement_background = {}  # depth before each grasp, including existing contents
    self.per.arm_points = lambda: self._points
    self._surface_pts = None
    self.per.arm_surface = lambda: self._surface_pts
    self.thread = threading.Thread(target=self._loop, name="rgbd-scene", daemon=True)
    self.detector = threading.Thread(target=self._detect_loop, name="rgbd-detector", daemon=True)
    self.thread.start()
    self.detector.start()

  def now(self):
    return time.monotonic()

  def set_joints(self, q6):
    with self.lock:
      self.q6 = np.array(q6).copy()
      self.q_hist.append((self.now(), self.q6))
      if len(self.q_hist) > 200:
        del self.q_hist[:100]

  def snapshot(self):
    with self.lock:
      return self._snapshot

  def _time(self, key, t0):
    v = self.timing.setdefault(key, [])
    v.append(self.now() - t0)
    if len(v) > 2000:
      del v[:1000]

  def wait_ready(self, timeout=15):
    end = self.now() + timeout
    while self.now() < end:
      s = self.snapshot()
      if s.error:
        raise SceneUnavailable(s.error)
      try:
        for n in self.names:fresh_object(s,n,self.now())
        return s
      except SceneUnavailable:
        pass
      self.stop.wait(0.05)
    raise SceneUnavailable(f"could not freshly detect all requested objects: {self.names}")

  def forget(self, name: str) -> None:
    """The object is now in the gripper: drop its track and stop detecting it until release."""
    with self._det_lock:
      self._forget.add(name)
      self._held.add(name)
    with self.lock:
      self._fixtures.pop(name,None)  # a former destination is now a moving payload

  def released(self, name: str) -> None:
    """The gripper opened: the object may be detected (at its new place) again."""
    with self._det_lock:
      self._held.discard(name)

  def inspect_target(self,name):
    """Establish destination interior/support from current depth before any grasp."""
    from so101_reach.manipulation.destination import destination_geometry,copy_detection
    s=self.snapshot();d=copy_detection(fresh_object(s,name,self.now()))
    with self.lock:
      last=getattr(self,"_last_depth",None)
    if last is None or self.now()-last[2]>.8:
      raise SceneUnavailable("no fresh depth for destination inspection")
    geo=destination_geometry(last[0],last[1],d,self.table_z)
    if geo is None:
      raise SceneUnavailable(f"{name!r}: interior/support not sufficiently visible")
    d.extra.update(geo)
    d.top_z=geo["rim_z"]
    d.center[2]=(d.top_z+d.support_z)/2
    return d

  def pin_target(self,name,detection):
    from so101_reach.manipulation.destination import copy_detection
    with self.lock:
      self._fixtures[name]=copy_detection(detection)

  def measure_placed(self,target,payload):
    from so101_reach.manipulation.destination import release_geometry
    with self.lock:
      last=getattr(self,"_last_depth",None)
      baseline=self._placement_background.get(target.name)
    if last is None or self.now()-last[2]>.8:return None
    return release_geometry(last[0],last[1],target,payload,self.table_z,baseline_depth=baseline)

  def placement_point(self,target,payload,occupants=()):
    from so101_reach.manipulation.destination import free_release_point
    with self.lock:last=getattr(self,"_last_depth",None)
    if last is None or self.now()-last[2]>.8:raise SceneUnavailable("no fresh depth for placement space")
    point=free_release_point(last[0],last[1],target,payload,occupants)
    with self.lock:self._placement_background[target.name]=last[0].copy()
    return point

  def measure_at(self, center: np.ndarray, size) -> dict | None:
    """Fresh depth geometry in the image region of an expected object box (placement check
    without detector identity). None if the region is hidden or has no valid depth."""
    with self.lock:
      last = getattr(self, "_last_depth", None)
      stamp = self._snapshot.stamp if last is None else last[2]
    if last is None:
      return None
    depth, v = last[:2]
    half = np.array(size, float) / 2 * 1.3
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half + center
    px = [v.project(c) for c in corners]
    if any(p is None for p in px):
      return None
    px = np.array(px, float)
    box = np.array([*px.min(0), *px.max(0)])
    m = measure_object(depth, (v.fx, v.fy, v.cx, v.cy), v.R, v.t, box, self.table_z)
    if m is None:
      return None
    return {**m, "age_s": self.now() - stamp}

  def vacated(self, ob: Detection) -> dict:
    """Is the support surface visible again where `ob` stood? Uses this worker's latest masked
    depth (arm pixels removed). Returns visible and vacated fractions of footprint samples."""
    with self.lock:
      last = getattr(self, "_last_depth", None)
      stamp = self._snapshot.stamp if last is None else last[2]
    if last is None:
      return {"visible": 0.0, "vacated": 0.0, "age_s": None}
    depth, v = last[:2]
    a = np.radians(ob.yaw_deg)
    axes = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    half = np.maximum(np.array(ob.size[:2]) / 2 - 0.003, 0.002)
    g = np.stack(np.meshgrid(np.linspace(-1, 1, 7), np.linspace(-1, 1, 7)), -1).reshape(-1, 2) * half
    xy = ob.center[:2] + g @ axes
    vis = vac = 0
    for p in np.column_stack((xy, np.full(len(xy), ob.support_z))):
      px = v.project(p)
      if px is None or not (0 <= px[1] < depth.shape[0] and 0 <= px[0] < depth.shape[1]):
        continue
      z = depth[px[1], px[0]]
      if z <= 0:
        continue
      vis += 1
      q = v.R @ np.array([(px[0] - v.cx) / v.fx * z, (px[1] - v.cy) / v.fy * z, z]) + v.t
      vac += abs(q[2] - ob.support_z) < 0.006
    n = len(xy)
    return {"samples": n, "visible_n": vis, "visible": vis / n, "vacated": (vac / vis) if vis else 0.0, "age_s": self.now() - stamp}

  def wait_fresh(self, max_age, timeout):
    """Block (the arm holds) until the published scene is younger than max_age."""
    end = self.now() + timeout
    while self.now() < end:
      s = self.snapshot()
      if s.error or self.now() - s.stamp <= max_age:
        return s
      self.stop.wait(0.01)
    return self.snapshot()

  def _identity_limit(self, name, occluded) -> float:
    return self.occluded_identity_max_age if name in occluded else self.identity_max_age

  def _pick_gate(self, name, d) -> bool:
    """A pick target must be graspable: within the width domain and within the arm's reach.
    Without this a second, unreachable instance ("the orange" at 0.55 m) or a larger same-colour
    object (the white box as "white cube") could take the name."""
    if name in self.max_width and min(d.size[:2]) > self.max_width[name]:
      self.gate_rejections[name] = f"{1000 * min(d.size[:2]):.0f} mm wide (graspable <= {1000 * self.max_width[name]:.0f} mm)"
      return False
    if name in self.max_reach:
      xy, radius = self.max_reach[name]
      r = float(np.linalg.norm(np.asarray(d.center[:2]) - xy))
      if r > radius:
        self.gate_rejections[name] = (f"out of reach at ({100 * d.center[0]:.0f}, {100 * d.center[1]:.0f}) cm: "
                                      f"{100 * r:.0f} cm from the shoulder axis (reach {100 * radius:.0f} cm)")
        return False
    return True

  def _in_workspace(self, c) -> bool:
    """Candidates centred more than 5 cm outside the configured site limits are not task objects
    (a white wall at y = 0.16 m was detected as 'white box' while the arm hid the box)."""
    if self.workspace is None:
      return True
    lo, hi = self.workspace
    return bool(np.all(c[:2] >= lo - 0.05) and np.all(c[:2] <= hi + 0.05))

  def _detect_hands(self, frame) -> list:
    """People's hands in the head view (Grounding DINO 'hand'), excluding the robot itself:
    boxes the projected arm silhouette covers, or whose depth lies on the arm. A hand pauses
    motion; it is never treated as a task object."""
    from so101_reach.agent.perception import View
    t0 = self.now()
    cam, calib = self.per.cams["head"]
    R, t = calib.pose()
    k = cam.intrinsics
    view = View(frame.color, R, t, k.fx, k.fy, k.cx, k.cy)
    mask = self.per._arm_mask(view, frame.depth.shape)
    arm = self._surface_pts
    out = []
    for score, box in self.per.model.detect(frame.color, "hand", self.hand_threshold, self.hand_threshold):
      x0, y0, x1, y1 = np.clip(np.rint(box).astype(int), 0, [frame.depth.shape[1], frame.depth.shape[0]] * 2)
      if x1 <= x0 or y1 <= y0 or (x1 - x0) * (y1 - y0) > 0.4 * frame.depth.size:
        continue
      if mask[y0:y1, x0:x1].mean() > 0.35:
        continue  # the gripper/arm itself
      center = None
      patch = frame.depth[y0 + (y1 - y0) // 4:y1 - (y1 - y0) // 4, x0 + (x1 - x0) // 4:x1 - (x1 - x0) // 4]
      zs = patch[patch > 0]
      if zs.size > 20:
        z = float(np.median(zs))
        u, v = (x0 + x1) / 2, (y0 + y1) / 2
        center = R @ np.array([(u - k.cx) / k.fx * z, (v - k.cy) / k.fy * z, z]) + t
        if arm is not None and len(arm) and np.min(np.linalg.norm(arm - center, axis=1)) < 0.03:
          continue  # depth on the arm itself
        if np.linalg.norm(center[:2]) > 0.65:
          continue  # outside the robot's reach area
      out.append({"bbox": [int(x0), int(y0), int(x1), int(y1)], "score": round(float(score), 3),
                  "center": None if center is None else center.round(3).tolist()})
    self._time("hands", t0)
    return out

  def _detect_loop(self):
    try:
      while not self.stop.is_set():
        if not self._det_wake.wait(0.1):
          continue
        self._det_wake.clear()
        with self._det_lock:
          req, self._det_request = self._det_request, None
        if req is None:
          continue
        frame_no, frame = req
        t0 = self.now()
        cands = {}
        for name in [n for n in self.names if n not in self._held]:
          t_n = self.now()
          cands[name] = [d for d in self.per.find(name, keep_ratio=0.0, frame=frame, remember=False)
                         if d.bbox is not None and self._in_workspace(d.center) and self._pick_gate(name, d)]
          self._time("find:" + name, t_n)
        chosen, info = assign_detections(cands, self.assign_margin)
        hands = self._detect_hands(frame) if self.watch_hands else []
        self._time("detect_total", t0)
        with self._det_lock:
          self._det_result = (frame_no, chosen, info)
          self._hands = (self.now(), hands)
    except Exception as e:
      with self.lock:
        self._snapshot = Snapshot(self.now(), {}, {}, error=f"detector failed: {e}")

  def _loop(self):
    from so101_reach.real.grasp_replay import ArmKinematics
    from scipy.spatial import cKDTree
    kin = ArmKinematics()  # worker owns its kinematics; never touches main-thread MjData
    surface = RobotSurface(kin.m)
    robot_pts = None
    ids = [kin.m.body(n).id for n in ("shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1", "camera_mount")]
    prev_gray, prev_arm, last_request, frame_no = None, None, -1e9, 0
    grays = {}  # recent grey frames by number, for propagating detector boxes
    tracks, objects, seen, confirmed, occluded = {}, {}, {}, {}, {}
    last_size = {}  # confirmed measured size per name (identity check on re-detection)
    try:
      while not self.stop.is_set():
        loop_start = self.now()
        for name in [n for n in tracks if self.now() - confirmed.get(n, -1e9) > self._identity_limit(n, occluded)]:
          tracks.pop(name, None)  # no good evidence for too long: wait for the detector
        views = self.per.views()
        captured = self.now()
        with self.lock:
          q = None if self.q6 is None else self.q6.copy()
          # Joint states over the frame's latency window: the image may show the arm up to
          # ~0.25 s ago, so the arm is removed wherever it was in that window.
          recent = [qq for tq, qq in self.q_hist if captured - self.latency_window <= tq <= captured]
        if q is not None:
          poses = (recent[::max(1, len(recent) // 4)] + [q]) if recent else [q]
          prox, dist = [], []
          for qq in poses:
            kin.fk(qq)
            prox.append(surface.world(kin.d, proximal=True))
            dist.append(surface.world(kin.d, proximal=False))
          self._points = np.concatenate([np.linspace(kin.d.xpos[a], kin.d.xpos[b], 16) for a, b in zip(ids[:-1], ids[1:])])
          robot_pts = (np.concatenate(prox), np.concatenate(dist))
          self._surface_pts = np.concatenate(robot_pts)
        self._time("grab", loop_start)
        v = views["head"]
        fr = self.per.frames["head"]
        gray = cv2.cvtColor(fr.color, cv2.COLOR_BGR2GRAY)
        frame_no += 1
        grays[frame_no] = gray
        for k in [k for k in grays if k < frame_no - 60]:
          del grays[k]
        arm_mask = self.per._arm_mask(v, fr.depth.shape)
        # Optical flow supplies fast box motion from features off the arm; depth is remeasured
        # below, never extrapolated. A box the arm hides keeps its place (it cannot be tracked).
        updated = {}
        static_only = set()
        if prev_gray is not None:
          for name, box in tracks.items():
            shift = flow_shift(prev_gray, gray, box, exclude=prev_arm)
            if shift is not None:
              updated[name] = box + np.tile(shift, 2)
            else:
              # Textureless or hidden: keep the box (static hypothesis). It only yields a fresh
              # observation if the depth re-measurement below agrees with the previous geometry.
              updated[name] = box
              static_only.add(name)
        tracks = updated
        # Adopt the newest detector result, propagated from its source frame to this one.
        with self._det_lock:
          result, self._det_result = self._det_result, None
          forget, self._forget = self._forget, set()
        for name in forget:
          for d in (tracks, objects, seen, confirmed, occluded):
            d.pop(name, None)
        held = set(self._held)
        if result is not None:
          src_no, chosen, info = result
          self.assignment = info
          for name in self.names:
            d = chosen.get(name)
            if name in held:
              continue
            if d is None:
              # No confident assignment this cycle (often occlusion by the arm). Identity carried by
              # continuous flow tracking survives only for identity_max_age after its last
              # confirmation; it is never re-established from an ambiguous detection.
              if name in tracks and self.now() - confirmed.get(name, -1e9) <= self._identity_limit(name, occluded):
                continue
              tracks.pop(name, None)
              objects.pop(name, None)
              continue
            if name in last_size and not size_consistent(d.size, last_size[name]):
              # Re-identification must agree with the object's confirmed size (the detector called
              # a white wall "white box" while the arm hid the box).
              self.rejected_identity.append((round(self.now(), 2), name, [round(x, 3) for x in d.size]))
              continue
            box = d.bbox.copy()
            if src_no != frame_no:
              shift = flow_shift(grays[src_no], gray, box) if src_no in grays else None
              if shift is None:
                tracks.pop(name, None)
                continue
              box = box + np.tile(shift, 2)
            if name in tracks:
              tb = tracks[name]
              inside = lambda c, b: b[0] <= c[0] <= b[2] and b[1] <= c[1] <= b[3]
              if not inside((box[:2] + box[2:]) / 2, tb) and not inside((tb[:2] + tb[2:]) / 2, box):
                tracks.pop(name, None)  # detector and tracker disagree: wait for agreement
                continue
            tracks[name] = box
            if name not in objects:
              objects[name] = d
            confirmed[name] = self.now()
        if now_due := (self.now() - last_request >= self.period):
          with self._det_lock:
            if self._det_request is None:
              self._det_request = (frame_no, fr)
              last_request = self.now()
              self._det_wake.set()
        t_m = self.now()
        depth = np.where(arm_mask, 0, fr.depth)
        trees = None if robot_pts is None else (cKDTree(robot_pts[0]), cKDTree(robot_pts[1]))
        def own_points(pts, trees=trees):
          if trees is None:
            return np.zeros(len(pts), bool)
          return np.isfinite(trees[0].query(pts, distance_upper_bound=self.proximal_pad)[0]) | \
                 np.isfinite(trees[1].query(pts, distance_upper_bound=self.self_pad)[0])
        with self.lock:
          self._last_depth = (depth, v, captured)
          fixtures=dict(self._fixtures)
        for name,d in fixtures.items():
          if name not in held:
            tracks[name]=d.bbox.copy() if d.bbox is not None else tracks.get(name)
            objects[name]=d
        status = {n: "untracked" for n in self.names}
        for name, box in list(tracks.items()):
          if box is None:
            continue
          # A box mostly hidden by the arm is not re-measured. Its last full measurement is fresh
          # again only if this frame's visible points agree with it (partial verification).
          x0,y0,x1,y1 = box.astype(int)
          covered = arm_mask[max(0,y0):max(0,y1),max(0,x0):max(0,x1)]
          if not covered.size:
            continue
          if name in fixtures:
            # Track the established fixture by positive current depth evidence. A payload
            # merging with its detector box must not alter its centre, rim, or object type.
            old=fixtures[name]
            pstats={}
            verdict="hidden" if covered.mean()>.9 else partial_consistent(depth,v,box,old,robot=own_points,stats=pstats)
            status[name]=f"fixture_{verdict}:{covered.mean():.2f} {pstats}"
            if verdict=="consistent":
              objects[name]=replace(old,extra={**old.extra,"fixture_verified":True})
              seen[name]=confirmed[name]=captured
              occluded.pop(name,None)
            elif verdict=="hidden" and covered.mean()>.35:
              occluded.setdefault(name,captured)
            else:
              occluded.pop(name,None)
              seen.pop(name,None)
            continue
          if covered.mean() > 0.08:
            old = objects.get(name)
            if old is None:
              continue
            pstats = {}
            verdict = "hidden" if covered.mean() > 0.9 else partial_consistent(depth, v, box, old, robot=own_points, stats=pstats)
            status[name] = f"{verdict}:{covered.mean():.2f} {pstats}"
            if verdict == "consistent":
              objects[name] = Detection(old.id,name,old.score,old.center,old.top_z,old.support_z,old.size,old.yaw_deg,"head",box.copy(),
                                        {**{k:x for k,x in old.extra.items() if k!="partial"},"partial":round(1-float(covered.mean()),2)})
              seen[name] = captured
              confirmed[name] = captured
              occluded.pop(name, None)
            elif verdict == "hidden":
              occluded.setdefault(name, captured)
            else:
              occluded.pop(name, None)  # visible and disagreeing: treat as changed, not hidden
            continue
          occluded.pop(name, None)
          m = measure_object(depth, (v.fx,v.fy,v.cx,v.cy), v.R,v.t,box,self.table_z)
          if m is None:
            status[name] = "no_depth"
            continue
          if max(m["size"][:2]) > 0.08:
            m = self.per._measure_large(depth,fr,box,m)
          extra={}
          # An open container needs geometric evidence of a lower interior, not just its name.
          if min(m["size"][:2])>0.06:
            bx0,by0,bx1,by1=box
            vv0,uu0=np.mgrid[int(max(0,by0+(by1-by0)*.35)):int(min(depth.shape[0],by1-(by1-by0)*.35)):3,
                            int(max(0,bx0+(bx1-bx0)*.35)):int(min(depth.shape[1],bx1-(bx1-bx0)*.35)):3]
            zz0=depth[vv0,uu0];valid0=zz0>0
            if valid0.sum()>8:
              inside=np.column_stack(((uu0[valid0]-v.cx)/v.fx*zz0[valid0],(vv0[valid0]-v.cy)/v.fy*zz0[valid0],zz0[valid0]))@v.R.T+v.t
              # Points above the rim are occluders in front of the interior (a payload carried over
              # the box turned it into "not a container" on hardware), not the interior itself.
              inside=inside[inside[:,2]<=m["top"]+0.005]
              if len(inside)>=30:
                extra["container"]=bool(m["top"]-np.median(inside[:,2])>0.012)
          old = objects.get(name)
          if "container" not in extra and old is not None and "container" in old.extra:
            extra["container"]=old.extra["container"]  # interior not measurable now: keep the last verdict
          if old is None:
            status[name] = "no_identity"
            continue
          if name in static_only and (np.linalg.norm(m["center"][:2]-old.center[:2])>0.012 or
                                      np.any(np.abs(np.array(m["size"])-np.array(old.size))>0.008)):
            status[name] = "static_mismatch"  # cannot track it and it no longer matches: stale
            continue
          if name in static_only:
            # Re-centre the kept box on the measured object (geometry closes the tracking loop).
            c = v.project(np.array([*m["center"][:2], m["top"]]))
            if c is not None:
              half = (box[2:] - box[:2]) / 2
              box = np.array([c[0]-half[0], c[1]-half[1], c[0]+half[0], c[1]+half[1]], float)
              tracks[name] = box
          objects[name] = Detection(old.id,name,old.score,m["center"],m["top"],m["support"],m["size"],m["yaw"],"head",box.copy(),extra)
          seen[name] = captured
          confirmed[name] = captured
          binding=self.per.instance_bindings.get(name)
          if binding is not None and np.linalg.norm(m['center'][:2]-binding[1][:2])<=.030:
            self.per.instance_bindings[name]=(binding[0],m['center'].copy())
          if name not in static_only or name not in last_size:
            last_size[name] = m["size"]
          status[name] = "measured"
        # Different requested names must not silently refer to the same physical box.
        for i, a in enumerate(self.names):
          for b in self.names[i+1:]:
            if a in objects and b in objects and np.linalg.norm(objects[a].center-objects[b].center)<0.015:
              seen.pop(a,None); seen.pop(b,None)
        self._time("measure", t_m);t_c=self.now()
        if not self.timeline or self.timeline[-1][1] != status:
          self.timeline.append((round(captured, 3), status))
          if len(self.timeline) > 400:del self.timeline[:200]
        vv, uu = np.mgrid[0:depth.shape[0]:6,0:depth.shape[1]:6]
        zz = depth[vv,uu]
        valid = (zz>0.08)&(zz<1.8)
        pc = np.column_stack(((uu[valid]-v.cx)/v.fx*zz[valid],(vv[valid]-v.cy)/v.fy*zz[valid],zz[valid])) @ v.R.T + v.t
        pc = pc[(pc[:,2]>self.table_z+0.004)&(np.linalg.norm(pc[:,:2],axis=1)<0.60)]
        if len(pc):
          _, idx = np.unique(np.round(pc/0.005).astype(int),axis=0,return_index=True)
          pc=pc[idx]
        if len(pc) and robot_pts is not None:
          # 3-D self-filter: the arm's own surface (over this frame's joint states) is not an
          # obstacle. Proximal links carry cable loops up to ~3 cm off the model (measured
          # 2026-11-07), so they get a wider pad; gripper/jaws keep the tight one. Obstacles
          # closer than the pad to the arm are not perceived (blind zone).
          own=own_points(pc)
          self.self_filtered=int(own.sum())
          pc=pc[~own]
        self._time("cloud", t_c)
        self.loop_s.append(self.now()-loop_start)
        if len(self.loop_s)>2000:del self.loop_s[:1000]
        with self.lock:
          hand_t, hand_list = self._hands
          self._snapshot = Snapshot(captured,dict(objects),dict(seen),pc,fr.color.copy(),frame_no,
                                    occluded=dict(occluded),joints=None if q is None else q.copy(),
                                    hands=list(hand_list) if self.now()-hand_t<=1.5 else [])
        prev_gray=gray
        prev_arm=arm_mask
    except Exception as e:
      with self.lock:
        self._snapshot=Snapshot(self.now(),{}, {}, error=f"camera worker failed: {e}")

  def close(self):
    self.stop.set()
    self._det_wake.set()
    self.thread.join(timeout=10)
    self.detector.join(timeout=10)
