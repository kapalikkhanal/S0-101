"""Rendered RGB-D stand-in for the head D435i, to exercise the real camera worker in simulation.

The physics (main) thread renders colour + depth into a buffer at a fixed rate; the
`CameraScene` worker thread blocks on `read()` exactly as on a RealSense pipeline and never
touches MjData. The simulation is paced to wall-clock time, because the worker's freshness
checks use time.monotonic(). Images are noise-free and perfectly calibrated: this validates
code paths (detection, flow, depth remeasurement, arm masking, obstacle cloud, camera-mode
collision checking, lift/placement evidence), not real sensor accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass
import threading
import time

import cv2
import mujoco
import numpy as np

from so101_reach.agent.sim_world import HEAD_SIZE, WRIST_SIZE
from so101_reach.real.cameras import Frame


@dataclass
class _Intrinsics:
  width: int
  height: int
  fx: float
  fy: float
  cx: float
  cy: float


class _FixedPose:
  def __init__(self, R, t):
    self.R, self.t = R, t

  def pose(self):
    return self.R.copy(), self.t.copy()


class RenderedRGBD:
  """`RGBDCamera` look-alike rendered from SimWorld. Fixed (head) camera only."""

  def __init__(self, world, cam: str = "head", fps: float = 15.0):
    self.world, self.name, self.period = world, cam, 1.0 / fps
    h, w = HEAD_SIZE if cam == "head" else WRIST_SIZE
    R, t, fx, fy, cx, cy = world.camera_model(cam)
    self.intrinsics = _Intrinsics(w, h, fx, fy, cx, cy)
    self.calib = _FixedPose(R, t)
    self._color = mujoco.Renderer(world.model, h, w)
    self._depth = mujoco.Renderer(world.model, h, w)
    self._depth.enable_depth_rendering()
    self._cond = threading.Condition()
    self._frame: Frame | None = None
    self._seq = 0
    self._read_seq = {}
    self._last_capture = -1e9
    self.closed = False
    self.frames_rendered = 0
    self.frames_reserved = 0

  # --- main (physics) thread -------------------------------------------------------------
  def capture(self, force: bool = False) -> None:
    now = float(self.world.data.time)
    if not force and now - self._last_capture < self.period - 1e-9:
      return
    self._last_capture = now
    self._color.update_scene(self.world.data, camera=self.name)
    color = cv2.cvtColor(self._color.render(), cv2.COLOR_RGB2BGR)
    self._depth.update_scene(self.world.data, camera=self.name)
    depth = self._depth.render().astype(np.float32)
    depth[(depth < 0.07) | (depth > 3.0)] = 0.0  # D435i-like valid range; 0 = no data
    with self._cond:
      self._frame = Frame(color, depth, time.time())
      self._seq += 1
      self.frames_rendered += 1
      self._cond.notify_all()

  # --- worker thread (RealSense interface) -----------------------------------------------
  def read(self, timeout_ms: int = 3000) -> Frame:
    """Next rendered frame. While the physics thread computes (simulated time frozen) the last
    render is still an exact image of the world, so it is re-served at the camera rate."""
    me = threading.get_ident()
    with self._cond:
      new = self._cond.wait_for(lambda: self.closed or self._seq > self._read_seq.get(me, 0), self.period)
      if self.closed:
        raise RuntimeError("rendered camera closed")
      if self._frame is None:
        if not self._cond.wait_for(lambda: self.closed or self._frame is not None, timeout_ms / 1000):
          raise RuntimeError(f"no rendered frame within {timeout_ms} ms")
      self._read_seq[me] = self._seq
      if new and self._frame.stamp >= time.time() - self.period:
        return self._frame
      self.frames_reserved += 1
      return Frame(self._frame.color, self._frame.depth, time.time())

  def warmup(self, seconds: float = 0.0) -> None:
    pass

  def depth_at(self, depth: np.ndarray, u: float, v: float, radius: int = 3) -> float:
    h, w = depth.shape
    u0, v0 = int(round(u)), int(round(v))
    patch = depth[max(v0 - radius, 0):min(v0 + radius + 1, h), max(u0 - radius, 0):min(u0 + radius + 1, w)]
    valid = patch[patch > 0]
    return float(np.median(valid)) if valid.size else float("nan")

  def deproject(self, u: float, v: float, z: float) -> np.ndarray:
    k = self.intrinsics
    return np.array([(u - k.cx) / k.fx * z, (v - k.cy) / k.fy * z, z])

  def close(self) -> None:
    with self._cond:
      self.closed = True
      self._cond.notify_all()
    self._color.close()
    self._depth.close()


class RealtimePacer:
  """Keeps simulated time from running ahead of wall time (the camera worker uses wall time)."""

  def __init__(self, world):
    self.world = world
    self.t0_wall = time.perf_counter()
    self.t0_sim = float(world.data.time)
    self.max_lag = 0.0

  def wait(self) -> None:
    ahead = (float(self.world.data.time) - self.t0_sim) - (time.perf_counter() - self.t0_wall)
    if ahead > 0:
      time.sleep(ahead)
    else:
      # Computation pauses freeze the world; continue in real time instead of fast-forwarding.
      self.max_lag = max(self.max_lag, -ahead)
      self.t0_wall, self.t0_sim = time.perf_counter(), float(self.world.data.time)
