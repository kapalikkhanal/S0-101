"""Find a container (box, bowl, tray) by text and measure it in the robot base frame.

Selection: the region Grounding DINO scores highest for the prompt. `lead` is its margin over
the next different region; below ~0.05 the name is ambiguous (e.g. "white box" scored the
white box 0.46 and the tan box - white inside - 0.43-0.45, while "white container" gave 0.59
vs 0.36). scripts/dino_view.py shows how DINO scores names on the live image.

Geometry (head-camera depth inside the chosen image box): points 1-25 cm above the table
are binned on a 1 cm grid, the largest connected patch is kept (drops the arm, cables and
the wall seen behind the box), its minimum-area rectangle is the footprint and the 90th
percentile height is the rim. Shallow containers (rim < 2 cm, e.g. a jar lid or a plate) have
too few raised points for that: their footprint is the image box projected onto the plane of
the rim (a rectangle around a round lid), the rim height the 95th percentile of the points
0.3-3 cm above the table (at least 8 mm).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from so101_reach.real.cameras import Frame, RGBDCamera
from so101_reach.real.objects import DinoModel
from so101_reach.real.orange import CameraPose


@dataclass
class Container:
  name: str
  center: np.ndarray  # (2,) footprint centre, base frame [m]
  size: np.ndarray  # (2,) footprint side lengths [m] (along `axes` columns)
  axes: np.ndarray  # (2, 2) unit side directions (columns)
  rim_z: float  # rim height, base frame [m]
  score: float  # prompt score
  margin: float  # lead over the next different region (0-1; small = ambiguous name)
  bbox: np.ndarray  # image box [px]

  def corners(self) -> np.ndarray:
    h = self.size / 2
    return np.array([self.center + self.axes @ (h * s) for s in ((1, 1), (-1, 1), (-1, -1), (1, -1))])

  def contains(self, xy: np.ndarray, grow: float = 0.0) -> bool:
    local = self.axes.T @ (np.asarray(xy)[:2] - self.center)
    return bool(np.all(np.abs(local) <= self.size / 2 + grow))

  def drop_point(self, toward: np.ndarray, margin: float) -> np.ndarray:
    """The point inside the footprint (at least `margin` from the walls) closest to `toward`
    (e.g. the arm's pan axis), so the arm reaches over the near wall as little as needed."""
    local = self.axes.T @ (np.asarray(toward)[:2] - self.center)
    half = np.maximum(self.size / 2 - margin, 0.0)
    return self.center + self.axes @ np.clip(local, -half, half)

  def describe(self) -> str:
    return (f"{self.name}: centre ({self.center[0]:.3f}, {self.center[1]:+.3f}) m, {self.size[0] * 100:.0f} x "
            f"{self.size[1] * 100:.0f} cm, rim z {self.rim_z * 100:+.1f} cm (score {self.score:.2f}, lead {self.margin:+.2f}"
            + (" - AMBIGUOUS, check with scripts/dino_view.py" if self.margin < 0.05 else "") + ")")


def _iou(a: np.ndarray, b: np.ndarray) -> float:
  x0, y0 = np.maximum(a[:2], b[:2])
  x1, y1 = np.minimum(a[2:], b[2:])
  inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
  return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def rank_regions(model: DinoModel, bgr: np.ndarray, prompt: str, threshold: float = 0.25,
                 same_iou: float = 0.5) -> list[tuple[float, float, np.ndarray]]:
  """[(lead, score, box)] for the prompt, best first: one entry per distinct region (overlapping
  boxes merged), lead = score minus the next region's score (the last one: its own score)."""
  regions: list[tuple[float, np.ndarray]] = []
  for s, b in model.detect(bgr, prompt, threshold, threshold):
    if all(_iou(b, rb) < same_iou for _, rb in regions):
      regions.append((s, b))
  return [(s - (regions[i + 1][0] if i + 1 < len(regions) else 0.0), s, b) for i, (s, b) in enumerate(regions)]


def _main_patches(pts: np.ndarray, cell: float, min_frac: float = 0.2) -> np.ndarray | None:
  """Points of all 8-connected patches (xy grid of `cell`) at least `min_frac` the size of the
  largest: a container's walls can be separate patches (side walls seen edge-on have no depth),
  while small clutter beside it is dropped."""
  if len(pts) < 30:
    return None
  lo = pts[:, :2].min(axis=0)
  ij = np.floor((pts[:, :2] - lo) / cell).astype(int)
  grid = np.zeros(ij.max(axis=0) + 1, np.uint8)
  grid[ij[:, 0], ij[:, 1]] = 1
  n, lab = cv2.connectedComponents(grid, connectivity=8)
  if n < 2:
    return None
  sizes = np.array([(lab == k).sum() for k in range(1, n)])
  keep = {k + 1 for k in np.nonzero(sizes >= min_frac * sizes.max())[0]}
  return pts[np.isin(lab[ij[:, 0], ij[:, 1]], list(keep))]


def _largest_patch(pts: np.ndarray, cell: float) -> np.ndarray | None:
  """Points of the largest 8-connected patch on a `cell` grid (xy)."""
  if len(pts) < 30:
    return None
  lo = pts[:, :2].min(axis=0)
  ij = np.floor((pts[:, :2] - lo) / cell).astype(int)
  grid = np.zeros(ij.max(axis=0) + 1, np.uint8)
  grid[ij[:, 0], ij[:, 1]] = 1
  n, lab = cv2.connectedComponents(grid, connectivity=8)
  if n < 2:
    return None
  sizes = [(lab == k).sum() for k in range(1, n)]
  return pts[lab[ij[:, 0], ij[:, 1]] == 1 + int(np.argmax(sizes))]


def measure(frame: Frame, cam: RGBDCamera, calib: CameraPose, table_z: float, bbox: np.ndarray, step: int = 2,
            cell: float = 0.01, band: tuple[float, float] = (0.01, 0.25)) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
  """(centre, size, axes, rim_z) of the raised structure inside the image box, or None."""
  R, t = calib.pose()
  x0, y0, x1, y1 = bbox.astype(int)
  vs, us = np.mgrid[max(y0, 0) : min(y1, frame.depth.shape[0]) : step, max(x0, 0) : min(x1, frame.depth.shape[1]) : step]
  z = frame.depth[vs, us]
  ok = z > 0
  if ok.sum() < 50:
    return None
  K = cam.intrinsics  # Pinhole (cm-level geometry; the colour stream's distortion is negligible here).
  zc = z[ok].astype(float)
  cam_pts = np.stack([(us[ok] - K.cx) / K.fx * zc, (vs[ok] - K.cy) / K.fy * zc, zc], axis=1)
  pts = cam_pts @ R.T + t
  h = pts[:, 2] - table_z
  pts = pts[(h > band[0]) & (h < band[1])]
  part = _main_patches(pts, cell)
  if part is None:
    return None
  rim = float(np.percentile(part[:, 2], 90))
  # Footprint from the upper half of the walls only: flat clutter beside / behind the box
  # (cables, tools, the floor of the wall's skirting) stays below it.
  upper = pts[(pts[:, 2] - table_z > 0.5 * (rim - table_z)) & (pts[:, 2] < rim + 0.02)]
  part = _main_patches(upper, cell)
  if part is None:
    return None
  (cx, cy), (w, l), ang = cv2.minAreaRect(part[:, :2].astype(np.float32))
  a = np.radians(ang)
  axes = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
  return np.array([cx, cy]), np.array([w, l]), axes, rim


def measure_flat(frame: Frame, cam: RGBDCamera, calib: CameraPose, table_z: float, bbox: np.ndarray, step: int = 2,
                 cap: float = 0.03) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
  """(centre, size, axes, rim_z) of a shallow container: the image box on the rim plane."""
  R, t = calib.pose()
  K = cam.intrinsics
  x0, y0, x1, y1 = bbox
  # Rim height from the central half of the image box.
  cu, cv_, hu, hv = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 4, (y1 - y0) / 4
  vs, us = np.mgrid[int(cv_ - hv) : int(cv_ + hv) : step, int(cu - hu) : int(cu + hu) : step]
  vs, us = vs.clip(0, frame.depth.shape[0] - 1), us.clip(0, frame.depth.shape[1] - 1)
  z = frame.depth[vs, us].astype(float)
  ok = z > 0
  cam_pts = np.stack([(us[ok] - K.cx) / K.fx * z[ok], (vs[ok] - K.cy) / K.fy * z[ok], z[ok]], axis=1)
  h = (cam_pts @ R.T + t)[:, 2] - table_z
  h = h[(h > 0.003) & (h < cap)]
  rim = table_z + float(np.clip(np.percentile(h, 95), 0.008, cap) if len(h) >= 20 else 0.012)
  pts = []
  for u, v in ((x0, y0), (x1, y0), (x1, y1), (x0, y1), (cu, y0), (cu, y1), (x0, cv_), (x1, cv_)):
    d = R @ np.array([(u - K.cx) / K.fx, (v - K.cy) / K.fy, 1.0])
    pts.append((t + (rim - t[2]) / d[2] * d)[:2])
  (cx, cy), (w, l), ang = cv2.minAreaRect(np.array(pts, np.float32))
  a = np.radians(ang)
  return np.array([cx, cy]), np.array([w, l]), np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]), rim


def find_container(cam: RGBDCamera, calib: CameraPose, table_z: float, prompt: str, frames: int = 5,
                   model: DinoModel | None = None) -> Container | None:
  """Detect and measure the container over a few frames (median); None if not found."""
  model = model or DinoModel.shared()
  found = []
  for _ in range(frames):
    fr = cam.read()
    ranked = rank_regions(model, fr.color, prompt)
    if not ranked:
      continue
    margin, score, box = ranked[0]
    geo = measure(fr, cam, calib, table_z, box)
    if geo is None or geo[3] - table_z < 0.02:
      geo = measure_flat(fr, cam, calib, table_z, box)
    if geo is not None:
      found.append((margin, score, box, *geo))
  if not found:
    return None
  ref = found[len(found) // 2]
  centers = np.array([f[3] for f in found])
  center = np.median(centers, axis=0)
  i = int(np.argmin(np.linalg.norm(centers - center, axis=1)))
  m, s, b, c, size, axes, rim = found[i]
  return Container(prompt, center, size, axes, float(np.median([f[6] for f in found])), s,
                   float(np.median([f[0] for f in found])), ref[2])


def draw_container(img: np.ndarray, cont: Container, cam: RGBDCamera, calib: CameraPose, drop: np.ndarray | None = None,
                   color: tuple[int, int, int] = (255, 0, 255), label: str | None = "PLACE: {name}") -> None:
  """Footprint at rim height (and the drop point) projected into the camera image, in place."""
  R, t = calib.pose()
  K = cam.intrinsics

  def proj(p: np.ndarray) -> tuple[int, int]:
    c = R.T @ (p - t)
    return int(K.fx * c[0] / c[2] + K.cx), int(K.fy * c[1] / c[2] + K.cy)

  pts = np.array([proj(np.r_[xy, cont.rim_z]) for xy in cont.corners()], np.int32)
  cv2.polylines(img, [pts], True, color, 2)
  if label:
    u, v = pts[:, 0].min(), pts[:, 1].min()
    cv2.putText(img, label.format(name=cont.name), (int(u), int(v) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
  if drop is not None:
    cv2.drawMarker(img, proj(np.r_[drop, cont.rim_z]), color, cv2.MARKER_CROSS, 18, 2)
