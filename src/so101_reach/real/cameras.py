"""RealSense RGB-D cameras of the SO-101 setup (librealsense, USB 3).

  head  : D435i (fixed, looking down at the table)
  wrist : D405  (clipped to the gripper)

Each read returns the BGR colour image and a depth image aligned to it (metres, 0 = no
data), so a colour pixel (u, v) maps directly to a 3-D point in the colour camera frame
(x right, y down, z forward) with `deproject`.

Note: the D435i only streams reliably on USB 3; on its USB 2 cable/port it enumerated but
delivered no frames from any sensor.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import pyrealsense2 as rs

CAMERAS = {"head": os.environ.get("SO101_HEAD_CAMERA_SERIAL", ""),
           "wrist": os.environ.get("SO101_WRIST_CAMERA_SERIAL", "")}
# The D405 produces colour on its stereo sensor, so colour and depth must share a resolution.
COLOR_MODES = {"head": (1280, 720), "wrist": (848, 480)}
DEPTH_MODES = {"head": (848, 480), "wrist": (848, 480)}


@dataclass
class Intrinsics:
  width: int
  height: int
  fx: float
  fy: float
  cx: float
  cy: float
  model: str
  coeffs: tuple[float, ...]

  @property
  def K(self) -> np.ndarray:
    return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

  @property
  def dist(self) -> np.ndarray:
    return np.array(self.coeffs[:5], dtype=np.float64)


@dataclass
class Frame:
  color: np.ndarray  # (H, W, 3) uint8 BGR
  depth: np.ndarray  # (H, W) float32 metres, aligned to colour; 0 = invalid
  stamp: float


class RGBDCamera:
  def __init__(self, name: str, width: int | None = None, height: int | None = None, fps: int = 30):
    self.name = name
    width, height = width or COLOR_MODES[name][0], height or COLOR_MODES[name][1]
    self.serial = CAMERAS[name]
    self.pipe = rs.pipeline()
    cfg = rs.config()
    if not self.serial:
      raise ValueError(f"Set SO101_{name.upper()}_CAMERA_SERIAL before opening this camera; see docs/HARDWARE.md")
    cfg.enable_device(self.serial)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    dw, dh = DEPTH_MODES[name]
    cfg.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, fps)
    prof = self.pipe.start(cfg)
    self.align = rs.align(rs.stream.color)
    self.depth_scale = prof.get_device().first_depth_sensor().get_depth_scale()
    self._rs_intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    i = self._rs_intr
    self.intrinsics = Intrinsics(i.width, i.height, i.fx, i.fy, i.ppx, i.ppy, str(i.model), tuple(i.coeffs))

  def read(self, timeout_ms: int = 3000) -> Frame:
    fs = self.align.process(self.pipe.wait_for_frames(timeout_ms))
    color = np.asanyarray(fs.get_color_frame().get_data()).copy()
    depth = np.asanyarray(fs.get_depth_frame().get_data()).astype(np.float32) * self.depth_scale
    return Frame(color, depth, time.time())

  def warmup(self, seconds: float = 1.5) -> None:
    t0 = time.time()
    while time.time() - t0 < seconds:
      self.read()

  def depth_at(self, depth: np.ndarray, u: float, v: float, radius: int = 3) -> float:
    """Median valid depth [m] in a (2r+1)^2 window around (u, v); nan if none."""
    h, w = depth.shape
    u0, v0 = int(round(u)), int(round(v))
    patch = depth[max(v0 - radius, 0) : min(v0 + radius + 1, h), max(u0 - radius, 0) : min(u0 + radius + 1, w)]
    valid = patch[patch > 0]
    return float(np.median(valid)) if valid.size else float("nan")

  def deproject(self, u: float, v: float, z: float) -> np.ndarray:
    """3-D point [m] in the colour camera frame for pixel (u, v) at depth z."""
    return np.array(rs.rs2_deproject_pixel_to_point(self._rs_intr, [float(u), float(v)], float(z)))

  def close(self) -> None:
    self.pipe.stop()

  def __enter__(self) -> RGBDCamera:
    return self

  def __exit__(self, *exc) -> None:
    self.close()


def open_camera(name: str, **kw) -> RGBDCamera:
  return RGBDCamera(name, **kw)
