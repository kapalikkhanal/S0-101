"""What the agent sees: named objects measured in 3-D, and annotated head / wrist images.

Objects are measured on whatever they rest on (table, a block, inside a box): from the depth
points inside the detection box, the top height is the 90th percentile of the central part,
the supporting surface the median height of a ring just outside the box, and the footprint
(size, yaw) the rotated rectangle of the points in the upper half of the object.

`RealPerception`: head + wrist RealSense, Grounding DINO. `SimPerception`: rendered views of
SimWorld with ground-truth detections (no DINO), for testing the agent loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Detection:
  id: int
  name: str
  score: float
  center: np.ndarray  # (3,) base frame [m]; z = middle of the object
  top_z: float  # [m] base frame
  support_z: float  # surface it rests on [m] base frame
  size: tuple[float, float, float]  # footprint length, width, height [m]
  yaw_deg: float  # direction of the footprint's long side
  camera: str
  bbox: np.ndarray | None = None  # head image box [px]
  extra: dict = field(default_factory=dict)

  def describe(self, table_z: float) -> str:
    l, w, h = self.size
    on = "table" if self.support_z - table_z < 0.01 else f"a surface {100 * (self.support_z - table_z):.1f} cm high"
    note = f" [{self.extra['note']}]" if self.extra.get("note") else ""
    return (f"#{self.id} {self.name}{note} (score {self.score:.2f}): centre x={100 * self.center[0]:.1f} y={100 * self.center[1]:.1f} cm, "
            f"top at h={100 * (self.top_z - table_z):.1f} cm, size {100 * l:.1f} x {100 * w:.1f} x {100 * h:.1f} cm, "
            f"long side yaw {self.yaw_deg:.0f} deg, resting on {on}")


@dataclass
class View:
  """A camera image with the pinhole model to project base-frame points into it."""

  image: np.ndarray
  R: np.ndarray  # base <- camera (OpenCV axes)
  t: np.ndarray
  fx: float
  fy: float
  cx: float
  cy: float

  def project(self, p: np.ndarray) -> tuple[int, int] | None:
    c = self.R.T @ (np.asarray(p) - self.t)
    if c[2] <= 0.01:
      return None
    return int(self.fx * c[0] / c[2] + self.cx), int(self.fy * c[1] / c[2] + self.cy)


def measure_object(depth: np.ndarray, K: tuple[float, float, float, float], R: np.ndarray, t: np.ndarray, bbox: np.ndarray,
                   table_z: float, step: int = 2) -> dict | None:
  """Top height, support height, centre and footprint of the object inside an image box."""
  fx, fy, cx, cy = K
  h, w = depth.shape
  x0, y0, x1, y1 = bbox

  def points(u0, v0, u1, v1):
    vs, us = np.mgrid[int(max(v0, 0)) : int(min(v1, h)) : step, int(max(u0, 0)) : int(min(u1, w)) : step]
    z = depth[vs, us].astype(float)
    ok = z > 0
    pc = np.stack([(us[ok] - cx) / fx * z[ok], (vs[ok] - cy) / fy * z[ok], z[ok]], axis=1)
    return pc @ R.T + t

  bw, bh = x1 - x0, y1 - y0
  inner = points(x0 + 0.25 * bw, y0 + 0.25 * bh, x1 - 0.25 * bw, y1 - 0.25 * bh)
  if len(inner) < 10:
    return None
  ring = np.concatenate([points(x0 - 0.3 * bw, y0 - 0.3 * bh, x1 + 0.3 * bw, y0), points(x0 - 0.3 * bw, y1, x1 + 0.3 * bw, y1 + 0.3 * bh),
                         points(x0 - 0.3 * bw, y0, x0, y1), points(x1, y0, x1 + 0.3 * bw, y1)])
  # Drop points far above the bulk of the central region (e.g. a cable hanging over a box).
  inner = inner[inner[:, 2] <= np.median(inner[:, 2]) + 0.10]
  top = float(np.percentile(inner[:, 2], 90))
  support = float(np.median(ring[:, 2])) if len(ring) >= 10 else table_z
  support = min(max(support, table_z), top - 0.005)
  allp = points(x0, y0, x1, y1)
  upper = allp[(allp[:, 2] > support + 0.5 * (top - support)) & (allp[:, 2] < top + 0.01)]
  if len(upper) < 8:
    upper = inner
  (ux, uy), (a, b), ang = cv2.minAreaRect(upper[:, :2].astype(np.float32))
  if a < b:
    a, b, ang = b, a, ang + 90
  return {"center": np.array([ux, uy, (top + support) / 2]), "top": top, "support": support,
          "size": (float(a), float(b), float(top - support)), "yaw": float((ang + 90) % 180 - 90)}


def _color_mask(hsv: np.ndarray, color: str) -> np.ndarray | None:
  """Pixels of a named colour (OpenCV HSV, H 0-180); None for words that are not colours."""
  h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
  masks = {
    "red": ((h <= 10) | (h >= 170)) & (s >= 90) & (v >= 60),
    "orange": (h > 10) & (h <= 22) & (s >= 120) & (v >= 80),
    "yellow": (h > 22) & (h <= 35) & (s >= 90) & (v >= 80),
    "green": (h > 35) & (h <= 85) & (s >= 60) & (v >= 40),
    "blue": (h > 90) & (h <= 130) & (s >= 70) & (v >= 40),
    "purple": (h > 130) & (h < 170) & (s >= 60) & (v >= 40),
    "pink": ((h >= 150) | (h <= 10)) & (s >= 30) & (s < 150) & (v >= 120),
    "white": (s <= 80) & (v >= 150),  # was 50: the head camera tints white cyan (S ~67)
    "black": v <= 70,
    "gray": (s <= 40) & (v > 70) & (v < 190),
    "grey": (s <= 40) & (v > 70) & (v < 190),
    "brown": (h >= 5) & (h <= 25) & (s >= 60) & (v >= 40) & (v <= 170),
  }
  return masks.get(color)


def color_fraction(bgr: np.ndarray, box: np.ndarray, name: str) -> float | None:
  """Share of the central half of the box with the colour named in `name` (None: no colour word)."""
  colors = color_words(name)
  if not colors:
    return None
  x0, y0, x1, y1 = box.astype(int)
  bw, bh = x1 - x0, y1 - y0
  if "rim" in name.lower().split() or "border" in name.lower().split():
    crop=bgr[max(0,y0):max(0,y1),max(0,x0):max(0,x1)]
    if not crop.size:return 0.0
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    ring=np.ones(crop.shape[:2],bool)
    hh,ww=ring.shape
    ring[hh//4:hh-hh//4,ww//4:ww-ww//4]=False
    return float(max(_color_mask(hsv,c)[ring].mean() for c in colors))
  crop = bgr[max(0, y0 + bh // 4) : max(0, y1 - bh // 4), max(0, x0 + bw // 4) : max(0, x1 - bw // 4)]
  if crop.size == 0:
    return 0.0
  hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
  return float(max(_color_mask(hsv, c).mean() for c in colors))


def color_words(name: str) -> list[str]:
  words=name.lower().split()
  # In 'brown orange', orange is the fruit noun, not a second colour constraint.
  if words and words[-1] in ("orange","oranges"):words=words[:-1]
  return [w for w in words if _color_mask(np.zeros((1,1,3),np.uint8),w) is not None]


def annotate(view: View, dets: list[Detection], tip: np.ndarray, table_z: float, title: str, axes: bool = False) -> np.ndarray:
  img = view.image.copy()
  for d in dets:
    corners = d.extra.get("corners")
    c2 = view.project(d.center)
    if corners is not None:
      pts = [view.project(p) for p in corners]
      pts = np.array([p for p in pts if p is not None], np.int32)
      if len(pts) >= 3:
        x, y, w, h = cv2.boundingRect(pts)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    elif d.bbox is not None and view is not None and d.camera == title.split()[0]:
      b = d.bbox.astype(int)
      cv2.rectangle(img, tuple(b[:2]), tuple(b[2:]), (0, 255, 0), 2)
    if c2 is not None:
      cv2.drawMarker(img, c2, (0, 255, 0), cv2.MARKER_TILTED_CROSS, 14, 2)
      cv2.putText(img, f"#{d.id} {d.name} top {100 * (d.top_z - table_z):.1f}cm", (c2[0] + 8, c2[1] - 8),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
  p = view.project(tip)
  if p is not None:
    cv2.circle(img, p, 9, (0, 0, 255), 2)
    cv2.putText(img, "TIP", (p[0] + 10, p[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
  if axes:  # +x / +y arrows on the table, 5 cm long, near the robot
    o = np.array([0.15, -0.20, table_z])
    po = view.project(o)
    for v, lab in ((np.array([0.05, 0, 0]), "+x"), (np.array([0, 0.05, 0]), "+y")):
      pe = view.project(o + v)
      if po and pe:
        cv2.arrowedLine(img, po, pe, (255, 255, 0), 2, tipLength=0.25)
        cv2.putText(img, lab, pe, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
  cv2.putText(img, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
  return img


class SimPerception:
  """Ground-truth 'detector' on SimWorld: a name matches a label if all its words occur in it
  (or its last word does, e.g. 'cube' -> 'red cube')."""

  def __init__(self, world):
    self.world = world
    self.table_z = world.cfg.table_z

  def _match(self, query: str, label: str, loose: bool = False) -> bool:
    q = query.lower().split()
    return all(w in label for w in q) or bool(loose and q and q[-1] in label)

  def find(self, name: str) -> list[Detection]:
    out = []
    labels = self.world.labels + [s["label"] for s in self.world.surfaces()]
    loose = not any(self._match(name, lab) for lab in labels)  # 'cube' matches every cube, 'red cube' only the red one
    for label in self.world.labels:
      if self._match(name, label, loose):
        g = self.world.object_truth(label)
        out.append(Detection(0, label, 1.0, g["center"], g["top"], g["bottom"], tuple(g["size"]), g["yaw_deg"], "sim",
                             extra={"corners": g["corners"]}))
    for s in self.world.surfaces():
      if self._match(name, s["label"], loose):
        out.append(Detection(0, s["label"], 1.0, s["center"], s["top"], self.table_z, tuple(s["size"]), s["yaw_deg"], "sim",
                             extra={"corners": s["corners"], "container": s.get("container", False)}))
    return out

  def views(self) -> dict[str, View]:
    out = {}
    for cam in ("head", "wrist"):
      R, t, fx, fy, cx, cy = self.world.camera_model(cam)
      out[cam] = View(self.world.render(cam), R, t, fx, fy, cx, cy)
    return out


class RealPerception:
  """Head RealSense + Grounding DINO for 3-D; the wrist camera only supplies its image.

  Measurements come from the head camera (the wrist camera's depth is invalid closer than ~7 cm
  and its calibration is ~1 cm off, which put a 4 cm cube's top at 6 cm). The arm's silhouette is
  projected into the head image: its pixels are never measured, detections it covers by more than
  a third (or that lie on the arm, e.g. its red cable) are dropped;
  an object hidden by the arm is reported from memory (its last good measurement, marked). Of
  several candidates for a name, those under `keep_ratio` x the best score are dropped, and for a
  name with a colour word at least `min_color` of the box centre must have that colour (DINO
  confuses colour words: 'red cube' scored 0.51 on a white box next to 0.82 on the red cube).
  """

  def __init__(self, head, wrist, head_calib, wrist_pose, table_z: float, threshold: float = 0.3,
               keep_ratio: float = 0.7, arm_points=None, min_color: float = 0.12, memory_frames: int = 3):
    from so101_reach.real.objects import DinoModel

    self.cams = {"head": (head, head_calib)}
    if wrist is not None:
      self.cams["wrist"] = (wrist, wrist_pose)
    self.table_z = table_z
    self.threshold, self.keep_ratio, self.min_color = threshold, keep_ratio, min_color
    self.arm_points = arm_points  # () -> (N, 3) points on the arm, base frame
    self.arm_surface = None  # optional () -> (N, 3) robot surface points (accurate silhouette)
    self.surface_pad_m = 0.008
    self.memory_frames = memory_frames
    self.model = DinoModel.shared()
    self.instance_bindings = {}
    self.frames: dict = {}
    self.frame_no = 0
    self.memory: dict[str, tuple[int, list[Detection]]] = {}

  def _grab(self) -> None:
    self.frames = {name: cam.read() for name, (cam, _) in self.cams.items()}
    self.frame_no += 1

  def _arm_mask(self, view: View, shape: tuple[int, int], radius_m: float = 0.035) -> np.ndarray:
    """Head-image pixels covered by the arm. With `arm_surface` (points on the robot model's
    surfaces at the current joints) this is the projected silhouette dilated by `surface_pad_m`;
    otherwise the chain of link points drawn as a thick band (~`radius_m` either side)."""
    surface = self.arm_surface() if getattr(self, "arm_surface", None) is not None else None
    if surface is not None and len(surface):
      c = (surface - view.t) @ view.R
      c = c[c[:, 2] > 0.05]
      mask = np.zeros(shape, np.uint8)
      if len(c):
        u = np.rint(view.fx * c[:, 0] / c[:, 2] + view.cx).astype(int)
        v = np.rint(view.fy * c[:, 1] / c[:, 2] + view.cy).astype(int)
        ok = (u >= 0) & (u < shape[1]) & (v >= 0) & (v < shape[0])
        mask[v[ok], u[ok]] = 255
        r = int(np.ceil(view.fx * self.surface_pad_m / max(float(np.percentile(c[:, 2], 10)), 0.05)))
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
      return mask > 0
    mask = np.zeros(shape, np.uint8)
    pts = self.arm_points() if self.arm_points is not None else None
    if pts is None:
      return mask.astype(bool)
    for a, b in zip(pts[:-1], pts[1:], strict=True):
      pa, pb = view.project(a), view.project(b)
      if pa is None or pb is None:
        continue
      depth = float((view.R.T @ (a - view.t))[2])
      cv2.line(mask, pa, pb, 255, max(2, int(2 * radius_m * view.fx / max(depth, 0.05))))
    return mask > 0

  def find(self, name: str, keep_ratio: float | None = None, frame=None, remember: bool = True) -> list[Detection]:
    """Measured candidates for `name`; keep_ratio=0 returns every candidate above threshold
    (for joint assignment across several requested names). `frame` measures a given head
    frame instead of the latest grabbed one (used by a detector thread)."""
    keep_ratio = self.keep_ratio if keep_ratio is None else keep_ratio
    if frame is None and not self.frames:
      self._grab()
    cam, calib = self.cams["head"]
    fr = self.frames["head"] if frame is None else frame
    R, t = calib.pose()
    k = cam.intrinsics
    view = View(fr.color, R, t, k.fx, k.fy, k.cx, k.cy)
    arm = self._arm_mask(view, fr.depth.shape)
    depth = np.where(arm, 0, fr.depth)  # never measure the arm's own surface
    arm_xyz = self.arm_points() if self.arm_points is not None else None
    arm_xyz = np.zeros((0, 3)) if arm_xyz is None else arm_xyz
    found = []
    words=name.lower().split()
    binding=self.instance_bindings.get(name)
    query=binding[0] if binding is not None else name
    if "rim" in words or "border" in words:
      # Colour describes the rim, not the central surface. Ground the noun, then
      # validate the requested rim colour separately on the image border.
      query=next((w for w in ("plate","bowl","tray","cup","box","container") if w in words),name)
    for score, box in self.model.detect(fr.color, query, self.threshold, self.threshold):
      x0, y0, x1, y1 = box.astype(int)
      covered = arm[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
      if covered.size == 0 or covered.mean() > 0.35:
        continue  # Mostly the arm (or hidden behind it): not measurable now.
      frac = color_fraction(fr.color, box, name)
      if frac is not None and frac < self.min_color:
        continue  # Named colour absent (DINO calls a white box 'red cube' when the red one is hidden).
      m = measure_object(depth, (k.fx, k.fy, k.cx, k.cy), R, t, box, self.table_z)
      if m is None:
        continue
      if binding is not None and np.linalg.norm(m["center"][:2]-binding[1][:2])>.030:
        continue  # Never substitute another fruit when the chosen instance disappears.
      if len(arm_xyz) and np.min(np.linalg.norm(arm_xyz - m["center"], axis=1)) < 0.04:
        continue  # On the arm itself.
      if max(m["size"][:2]) > 0.08:
        m = self._measure_large(depth, fr, box, m)
      found.append(Detection(0, name, score, m["center"], m["top"], m["support"], m["size"], m["yaw"], "head", box))
    if found:
      best = max(d.score for d in found)
      found = [d for d in sorted(found, key=lambda d: -d.score) if d.score >= keep_ratio * best]
      if remember:
        self.memory[name] = (self.frame_no, found)
      return found
    if not remember:
      return []
    if name in self.memory:
      n, dets = self.memory[name]
      if self.frame_no - n > self.memory_frames:
        return []
      out = []
      for d in dets:
        d2 = Detection(0, d.name, d.score, d.center, d.top_z, d.support_z, d.size, d.yaw_deg, d.camera, d.bbox, dict(d.extra))
        d2.extra["note"] = f"not visible now (hidden by the arm?) - last measured {self.frame_no - n} observation(s) ago"
        out.append(d2)
      return out
    return []

  def _measure_large(self, depth: np.ndarray, fr, box: np.ndarray, m: dict) -> dict:
    """Large (often hollow) objects - boxes, trays, bowls: the object-centre estimate above is
    biased by inner walls (a 15 cm box measured 6-9 cm off and its rim at 1.8 cm). Use the
    container measurement instead: largest connected patch of raised points, rotated-rectangle
    footprint, rim height (shallow ones: image box on the rim plane)."""
    from types import SimpleNamespace

    from so101_reach.real.containers import measure, measure_flat

    cam, calib = self.cams["head"]
    frame = SimpleNamespace(depth=depth, color=fr.color)
    geo = measure(frame, cam, calib, self.table_z, box)
    if geo is None or geo[3] - self.table_z < 0.02:
      geo = measure_flat(frame, cam, calib, self.table_z, box)
    if geo is None:
      return m
    center, size, axes, rim = geo
    a, b = (size[0], size[1]) if size[0] >= size[1] else (size[1], size[0])
    long_axis = axes[:, 0] if size[0] >= size[1] else axes[:, 1]
    yaw = float((np.degrees(np.arctan2(long_axis[1], long_axis[0])) + 90) % 180 - 90)
    return {"center": np.array([center[0], center[1], (rim + self.table_z) / 2]), "top": float(rim), "support": self.table_z,
            "size": (float(a), float(b), float(rim - self.table_z)), "yaw": yaw}

  def views(self) -> dict[str, View]:
    self._grab()
    out = {}
    for name, (cam, calib) in self.cams.items():
      R, t = calib.pose()
      k = cam.intrinsics
      out[name] = View(self.frames[name].color, R, t, k.fx, k.fy, k.cx, k.cy)
    return out
