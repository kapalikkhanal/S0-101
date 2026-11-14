"""AprilTag detection for the tag on the SO-101 gripper (OpenCV ArUco module).

The installed tag is AprilTag family 16h5, id 0. 16h5 has only 30 codes, so detections are
accepted only for the expected id and a plausible size. OpenCV's default error correction
is kept: with exact-code-only decoding (errorCorrectionRate = 0) the 47 px tag on the
head camera was found in 2 % of frames, with the default in 99-100 % (no other ids seen).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

TAG_FAMILY = "DICT_APRILTAG_16h5"
TAG_ID = 0


@dataclass
class TagDetection:
  tag_id: int
  corners: np.ndarray  # (4, 2) pixels, OpenCV order (top-left, top-right, bottom-right, bottom-left)
  center: np.ndarray  # (2,)
  side_px: float
  rvec: np.ndarray | None = None  # tag pose in the camera frame (needs intrinsics + size)
  tvec: np.ndarray | None = None  # [m]


class TagDetector:
  def __init__(self, family: str = TAG_FAMILY, tag_id: int = TAG_ID, min_side_px: float = 15.0):
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.adaptiveThreshWinSizeMin = 5
    params.adaptiveThreshWinSizeMax = 45
    params.adaptiveThreshWinSizeStep = 4
    self.det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, family)), params)
    self.tag_id = tag_id
    self.min_side_px = min_side_px

  def detect(self, bgr: np.ndarray, K: np.ndarray | None = None, dist: np.ndarray | None = None,
             tag_size_m: float | None = None) -> TagDetection | None:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    corners, ids, _ = self.det.detectMarkers(gray)
    if ids is None:
      return None
    for c, i in zip(corners, ids.flatten(), strict=True):
      c = c.reshape(4, 2).astype(np.float64)
      side = float(np.mean([np.linalg.norm(c[k] - c[(k + 1) % 4]) for k in range(4)]))
      if int(i) != self.tag_id or side < self.min_side_px:
        continue
      d = TagDetection(int(i), c, c.mean(axis=0), side)
      if K is not None and tag_size_m is not None:
        h = tag_size_m / 2
        obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj, c, K, dist if dist is not None else np.zeros(5), flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if ok:
          d.rvec, d.tvec = rvec.ravel(), tvec.ravel()
      return d
    return None


def draw(bgr: np.ndarray, d: TagDetection | None) -> np.ndarray:
  out = bgr.copy()
  if d is None:
    cv2.putText(out, "tag not found", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return out
  cv2.polylines(out, [d.corners.astype(np.int32)], True, (0, 255, 0), 2)
  cv2.circle(out, tuple(d.center.astype(int)), 4, (0, 0, 255), -1)
  label = f"16h5 id {d.tag_id}  {d.side_px:.0f}px"
  if d.tvec is not None:
    label += f"  cam xyz {d.tvec[0]:+.3f} {d.tvec[1]:+.3f} {d.tvec[2]:+.3f} m"
  cv2.putText(out, label, (int(d.corners[:, 0].min()), int(d.corners[:, 1].min()) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
  return out
