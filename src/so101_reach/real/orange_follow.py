"""The real SO-101 hovers above an orange seen by the head (and wrist) camera and follows it.

Camera threads (~30 Hz each): detect the orange (stick to the same one when several are
visible), fuse the cameras' base-frame estimates, smooth, and publish a hover target
`hover` metres above its top.
Control loop (50 Hz): MouseFollowCore with that target instead of the mouse - same site
limits, minimum height, workspace projection and blocked-joint hold. Starts FROZEN; when
the orange is lost (e.g. hidden by the arm) the last target is held.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from so101_reach.real.cameras import RGBDCamera
from so101_reach.real.mouse_real import MouseFollowCore
from so101_reach.real.orange import OrangeDetection, OrangeDetector, draw


class _Source:
  def __init__(self, name: str, cam: RGBDCamera, det: OrangeDetector):
    self.name, self.cam, self.det = name, cam, det
    self.center: np.ndarray | None = None
    self.radius = 0.0
    self.stamp = 0.0
    self.image: np.ndarray | None = None
    self.dets: list[OrangeDetection] = []
    self.chosen: OrangeDetection | None = None
    self.fps = 0.0


class OrangeTracker:
  """Fuses orange detections from one or more cameras (head first, then e.g. the wrist).

  Each camera runs in its own thread; detections are associated with the tracked orange
  (sticky selection), fresh per-camera estimates (< `fresh` s old) are averaged, and the
  result is smoothed. The orange counts as lost when no camera has seen it for `lost_after`.
  """

  def __init__(
    self,
    sources: list[tuple[str, RGBDCamera, OrangeDetector]],
    hover: float = 0.05,
    smoothing: float = 0.4,
    lost_after: float = 0.5,
    fresh: float = 0.25,
    name: str = "orange",
  ):
    self.sources = [_Source(*s) for s in sources]
    self.name = name
    self.overlay = None  # optional callable(image) drawing extra marks on the head image
    self.hover = hover
    self.alpha = smoothing
    self.lost_after = lost_after
    self.fresh = fresh
    self.center: np.ndarray | None = None
    self.radius = 0.0
    self.last_seen = 0.0
    self.stop = threading.Event()
    self._lock = threading.Lock()
    self._candidate: np.ndarray | None = None
    self._candidate_n = 0
    self.threads = [threading.Thread(target=self._loop, args=(src,), daemon=True) for src in self.sources]

  def start(self) -> None:
    for t in self.threads:
      t.start()

  def _loop(self, src: _Source) -> None:
    n, t0 = 0, time.time()
    while not self.stop.is_set():
      fr = src.cam.read()
      dets = src.det.detect_all(fr)
      with self._lock:
        chosen = self._select(dets)
        if chosen is not None:
          src.center, src.radius, src.stamp = chosen.center_base, chosen.radius_m, time.time()
          self._fuse()
        src.chosen, src.dets, src.image = chosen, dets, fr.color
      n += 1
      if time.time() - t0 > 1.0:
        src.fps, n, t0 = n / (time.time() - t0), 0, time.time()

  def _fuse(self) -> None:
    now = time.time()
    fresh = [s for s in self.sources if s.center is not None and now - s.stamp < self.fresh]
    c = np.mean([s.center for s in fresh], axis=0)
    self.radius = float(np.mean([s.radius for s in fresh]))
    restart = self.center is None or now - self.last_seen > 1.0
    self.center = c if restart else (1 - self.alpha) * self.center + self.alpha * c
    self.last_seen = now

  def _select(self, dets: list[OrangeDetection]) -> OrangeDetection | None:
    """Stick to the tracked orange: accept a detection near the last position (10 cm while
    tracking, 15 cm for 3 s after a loss); a detection elsewhere must persist for ~0.5 s
    (15 frames within 3 cm) before the tracker switches to it."""
    if not dets:
      return None
    if self.center is None:
      return max(dets, key=lambda d: d.radius_px)
    best = min(dets, key=lambda d: np.linalg.norm(d.center_base - self.center))
    gap = time.time() - self.last_seen
    radius = 0.10 if gap < self.lost_after else 0.15
    if np.linalg.norm(best.center_base - self.center) < radius and gap < 3.0:
      self._candidate, self._candidate_n = None, 0
      return best
    new = max(dets, key=lambda d: d.radius_px)
    if self._candidate is not None and np.linalg.norm(new.center_base - self._candidate) < 0.03:
      self._candidate_n += 1
    else:
      self._candidate, self._candidate_n = new.center_base, 1
    if self._candidate_n >= 15:
      self._candidate, self._candidate_n = None, 0
      self.last_seen = 0.0  # Restart smoothing at the new object.
      return new
    return None

  @property
  def visible(self) -> bool:
    return self.center is not None and time.time() - self.last_seen < self.lost_after

  @property
  def seen_by(self) -> list[str]:
    now = time.time()
    return [s.name for s in self.sources if s.center is not None and now - s.stamp < self.fresh]

  @property
  def fps(self) -> float:
    return self.sources[0].fps

  def hover_target(self) -> np.ndarray | None:
    with self._lock:
      if not self.visible:
        return None
      return self.center + np.array([0.0, 0.0, self.radius + self.hover])

  def annotated(self, lines: list[str]) -> np.ndarray | None:
    """Head image with detections; further cameras as insets in the top-right corner."""
    with self._lock:
      main = self.sources[0]
      if main.image is None:
        return None
      out = draw(main.image, main.dets, main.chosen, lines)
      if self.overlay is not None:
        self.overlay(out)
      x = out.shape[1]
      for src in self.sources[1:]:
        if src.image is None:
          continue
        inset = draw(src.image, src.dets, src.chosen, [src.name])
        w = out.shape[1] // 3
        inset = cv2.resize(inset, (w, int(inset.shape[0] * w / inset.shape[1])))
        x -= w
        out[: inset.shape[0], x : x + w] = inset
        cv2.rectangle(out, (x, 0), (x + w - 1, inset.shape[0] - 1), (255, 255, 255), 2)
      return out


def status_lines(core: MouseFollowCore, tr: OrangeTracker) -> list[str]:
  o = tr.center
  t, p = core.target, core.tcp
  return [
    f"{core.message}   speed {core.ctrl.cfg.speed_factor * 100:.0f}% (+/-)   hover {tr.hover * 100:.0f} cm (up/down)",
    f"{tr.name} " + (f"({o[0]:+.3f} {o[1]:+.3f} {o[2]:+.3f}) m" if o is not None else "-")
    + ("  seen by " + "+".join(tr.seen_by) if tr.visible else "  [LOST - holding]")
    + "   " + " ".join(f"{s.name} {s.fps:.0f} fps" for s in tr.sources),
    f"target ({t[0]:+.3f} {t[1]:+.3f} {t[2]:+.3f})  gripper ({p[0]:+.3f} {p[1]:+.3f} {p[2]:+.3f})  error {np.linalg.norm(t - p) * 1e3:.0f} mm  {core.note}",
  ]


def run_window(
  core: MouseFollowCore,
  tr: OrangeTracker,
  width: int = 1280,
  height: int = 720,
  step_fn=None,
  keys: dict | None = None,
  extra_status=None,
  title: str = "SO-101 REAL ARM - follows the orange (Esc: hold + quit)",
  hold_on_exit: bool = True,
) -> None:
  """Window with the annotated head (+ wrist inset) image and the 50 Hz control loop.

  step_fn: replaces the default control step (orange following); keys: extra
  {glfw key: callback}; extra_status: () -> str shown as the first status line;
  `toggle_fn` for F / click can be given as keys["F"].
  """
  import glfw
  import mujoco

  keys = keys or {}
  toggle = keys.pop("F", core.toggle)

  if not glfw.init():
    raise RuntimeError("GLFW init failed")
  win = glfw.create_window(width, height, title, None, None)
  glfw.make_context_current(win)
  glfw.swap_interval(0)
  model = mujoco.MjModel.from_xml_string("<mujoco/>")
  ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

  def on_key(w, key, sc, action, mods):
    if action != glfw.PRESS:
      return
    if key == glfw.KEY_ESCAPE:
      glfw.set_window_should_close(w, True)
    elif key == glfw.KEY_F:
      toggle()
    elif key in keys:
      keys[key]()
    elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD, glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
      step = 0.05 if key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD) else -0.05
      core.ctrl.cfg.speed_factor = float(np.clip(round(core.ctrl.cfg.speed_factor + step, 2), 0.05, 1.0))
    elif key in (glfw.KEY_UP, glfw.KEY_DOWN):
      tr.hover = float(np.clip(tr.hover + (0.01 if key == glfw.KEY_UP else -0.01), 0.02, 0.20))

  glfw.set_key_callback(win, on_key)
  glfw.set_mouse_button_callback(win, lambda w, b, a, m: toggle() if (b == glfw.MOUSE_BUTTON_LEFT and a == glfw.PRESS) else None)
  deadline = time.perf_counter()
  try:
    while not glfw.window_should_close(win):
      if step_fn is not None:
        step_fn()
      else:
        core.tick(tr.hover_target() if core.following else None)
      lines = status_lines(core, tr)
      img = tr.annotated(([extra_status()] if extra_status is not None else []) + lines)
      if img is not None:
        fb_w, fb_h = glfw.get_framebuffer_size(win)
        rgb = cv2.cvtColor(cv2.resize(img, (fb_w, fb_h)), cv2.COLOR_BGR2RGB)[::-1].copy()
        mujoco.mjr_drawPixels(rgb.ravel(), None, mujoco.MjrRect(0, 0, fb_w, fb_h), ctx)
        glfw.swap_buffers(win)
      glfw.poll_events()
      deadline += core.plant.dt
      delay = deadline - time.perf_counter()
      if delay > 0:
        time.sleep(delay)
      else:
        deadline = time.perf_counter()
  finally:
    if hold_on_exit:
      core.plant.hold()
    glfw.terminate()
