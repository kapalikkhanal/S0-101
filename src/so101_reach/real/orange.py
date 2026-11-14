"""Orange (fruit) detection in the head camera and its 3-D position in the robot base frame.

Colour: very saturated orange pixels (HSV hue 5-19 of 180, S >= 200, V >= 100). Measured on
this setup: orange H 13-18 / S 220-255, wooden table H 18-31 / S 38-188 (5th-95th pct), so
saturation separates fruit from wood under the camera's changing auto-exposure; red servo
cables (hue < 5) and pink objects (hue > 150) are excluded by hue.
Blobs are kept if large enough and round.

3-D: the radius comes from the blob's angular size and the surface depth (sphere geometry). If the orange rests on the
table (the usual case) its centre is the intersection of the camera ray through the blob
centre with the plane one radius above the table; this avoids the D435i depth bias on fruit
skin (measured ~14 mm too deep) and depth that hits the arm when it hovers above the fruit.
When there is no depth (closer than the sensor's minimum range, e.g. the wrist camera next
to the fruit) the size is derived from that ray/plane distance instead. Only with
`allow_lifted` is a raised orange located from depth (median depth of the blob's
inner disc + one radius along the ray). Blobs must be fruit-sized (radius 18-45 mm) and
fill >= 60 % of their enclosing circle (partly hidden slivers are rejected). The table height is
measured from depth over the whole image, near the base plane (`estimate_table_z`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from so101_reach.real.cameras import Frame, RGBDCamera

HSV_LO = np.array([5, 200, 100])
HSV_HI = np.array([19, 255, 255])


@dataclass
class OrangeDetection:
  pixel: np.ndarray  # (2,) blob centre [px]
  radius_px: float
  radius_m: float
  center_cam: np.ndarray  # (3,) camera frame [m]
  center_base: np.ndarray  # (3,) robot base frame [m]
  circularity: float
  on_table: bool = False
  surface_depth: float = float("nan")  # median depth of the blob's inner disc [m]
  label: str = ""  # detector label + score (open-vocabulary detectors)


class CameraPose(Protocol):
  def pose(self) -> tuple[np.ndarray, np.ndarray]:
    """Camera rotation (3x3) and position (3,) in the robot base frame, now."""
    ...


class OrangeDetector:
  def __init__(
    self,
    cam: RGBDCamera,
    calib: CameraPose,
    table_z: float | None = None,
    min_area_px: int = 250,
    min_circularity: float = 0.6,
    min_fill: float = 0.6,
    radius_range_m: tuple[float, float] = (0.018, 0.045),
    allow_lifted: bool = False,
  ):
    self.cam = cam
    self.calib = calib
    self.table_z = table_z
    self.min_fill = min_fill
    self.radius_range = radius_range_m
    self.allow_lifted = allow_lifted
    self.min_area = min_area_px
    self.min_circ = min_circularity
    self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    self.exclude: list = []  # areas (e.g. containers.Container) whose contents are ignored

  def mask(self, bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, HSV_LO, HSV_HI)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self.kernel)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, self.kernel)

  def candidates(self, frame: Frame) -> list[tuple[float, float, float, float, str]]:
    """Image blobs (u, v, radius_px, circularity, label): saturated-orange, large, round, filled."""
    out = []
    contours, _ = cv2.findContours(self.mask(frame.color), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
      area = cv2.contourArea(c)
      if area < self.min_area:
        continue
      perim = cv2.arcLength(c, True)
      circ = 4 * np.pi * area / max(perim**2, 1e-9)
      if circ < self.min_circ:
        continue
      (u, v), r_px = cv2.minEnclosingCircle(c)
      if area / (np.pi * r_px**2) < self.min_fill:
        continue  # Partly hidden (e.g. a sliver beside the arm): its depth / centre are unreliable.
      out.append((u, v, r_px, circ, ""))
    return out

  def detect_all(self, frame: Frame) -> list[OrangeDetection]:
    R, t = self.calib.pose()
    out = []
    for u, v, r_px, circ, label in self.candidates(frame):
      d = self.locate(frame, u, v, r_px, circ, R, t)
      if d is not None and any(area.contains(d.center_base, grow=0.01) for area in self.exclude):
        continue  # Already placed (inside a container).
      if d is not None:
        d.label = label
        out.append(d)
    return out

  def locate(self, frame: Frame, u: float, v: float, r_px: float, circ: float, R: np.ndarray, t: np.ndarray) -> OrangeDetection | None:
    """3-D centre of a roughly spherical object seen at pixel (u, v) with radius r_px."""
    fx = self.cam.intrinsics.fx
    disc = np.zeros(frame.depth.shape, np.uint8)
    cv2.circle(disc, (int(u), int(v)), max(int(r_px * 0.5), 2), 255, -1)
    z = frame.depth[(disc > 0) & (frame.depth > 0)]
    # Sphere seen at angular radius a (tan a = r_px / fx) with its front surface at z_surf:
    # sin a = r / (z_surf + r)  ->  r = z_surf sin a / (1 - sin a). (r_px z / fx under-
    # estimates close up, e.g. 21 vs 27 mm at 8 cm from the wrist camera.)
    sin_a = np.sin(np.arctan(r_px / fx))
    if z.size < 10:
      # No depth (closer than the sensor's minimum range, ~7 cm for the D405): locate it on the
      # table plane from the viewing ray alone and take the size from that distance.
      hit = self._ray_plane_sphere(u, v, sin_a, R, t)
      if hit is None:
        return None
      base, r_m = hit
      if not self.radius_range[0] <= r_m <= self.radius_range[1]:
        return None
      return OrangeDetection(np.array([u, v]), r_px, r_m, R.T @ (base - t), base, circ, True)
    z_surf = float(np.median(z))
    r_m = z_surf * sin_a / (1.0 - sin_a)
    if not self.radius_range[0] <= r_m <= self.radius_range[1]:
      return None  # Not fruit-sized (specks, reflections).
    surf = self.cam.deproject(u, v, z_surf)
    center = surf + r_m * surf / np.linalg.norm(surf)
    base = R @ center + t
    on_table = False
    if self.table_z is not None and (not self.allow_lifted or base[2] - self.table_z < 2.5 * r_m):
      hit = self._ray_plane(u, v, self.table_z + r_m, R, t)
      if hit is None:
        return None
      base, on_table = hit, True
    return OrangeDetection(np.array([u, v]), r_px, r_m, center, base, circ, on_table, z_surf)

  def _ray_plane(self, u: float, v: float, z: float, R: np.ndarray, o: np.ndarray) -> np.ndarray | None:
    d_cam = self.cam.deproject(u, v, 1.0)
    d = R @ (d_cam / np.linalg.norm(d_cam))
    if abs(d[2]) < 1e-6:
      return None
    t = (z - o[2]) / d[2]
    return o + t * d if t > 0 else None

  def _ray_plane_sphere(self, u: float, v: float, sin_a: float, R: np.ndarray, t: np.ndarray):
    """Depth-free: a sphere resting on the table, seen at angular radius a along the ray of
    pixel (u, v). Iterates centre height = table + r and r = distance * sin a."""
    if self.table_z is None:
      return None
    r = 0.026
    for _ in range(5):
      hit = self._ray_plane(u, v, self.table_z + r, R, t)
      if hit is None:
        return None
      r = float(np.linalg.norm(hit - t)) * sin_a
    return hit, r

  def estimate_table_z(self, frame: Frame, around: OrangeDetection | None = None, band: float = 0.05) -> float | None:
    """Table height in the base frame: depth points sampled over the whole image (excluding
    orange pixels), keeping only those within `band` of the base plane so the arm, boxes and
    other objects cannot bias it. None if fewer than 50 table points are visible."""
    del around
    mask = self.mask(frame.color)
    h, w = mask.shape
    R, t = self.calib.pose()
    zs = []
    for v in range(h // 20, h, h // 20):
      for u in range(w // 30, w, w // 30):
        if mask[v, u]:
          continue
        z = self.cam.depth_at(frame.depth, u, v, 2)
        if np.isfinite(z):
          pz = (R @ self.cam.deproject(u, v, z) + t)[2]
          if abs(pz) < band:
            zs.append(pz)
    return float(np.median(zs)) if len(zs) >= 50 else None

  def detect(self, frame: Frame, near: np.ndarray | None = None) -> OrangeDetection | None:
    """The largest orange, or the one closest to `near` (base frame) when tracking."""
    dets = self.detect_all(frame)
    if not dets:
      return None
    if near is not None:
      return min(dets, key=lambda d: np.linalg.norm(d.center_base - near))
    return max(dets, key=lambda d: d.radius_px)


def draw(bgr: np.ndarray, dets: list[OrangeDetection], chosen: OrangeDetection | None, text: list[str] | None = None) -> np.ndarray:
  out = bgr.copy()
  for d in dets:
    color = (0, 255, 0) if d is chosen else (0, 200, 255)
    cv2.circle(out, tuple(d.pixel.astype(int)), int(d.radius_px), color, 2)
    p = d.center_base
    cv2.putText(out, f"{d.label + '  ' if d.label else ''}({p[0]:+.3f} {p[1]:+.3f} {p[2]:+.3f}) m  r {d.radius_m * 1e3:.0f} mm",
                (int(d.pixel[0] - d.radius_px), int(d.pixel[1] - d.radius_px - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
  for i, t in enumerate(text or []):
    cv2.putText(out, t, (15, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
  return out
