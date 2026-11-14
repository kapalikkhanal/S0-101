"""Real SO-101 follows the mouse pointer (MuJoCo twin window + 50 Hz control loop).

Safety: starts FROZEN (holds pose) until left click / F; every cursor target is clipped to
the site limits (configs/real_workspace.json), the minimum height and the verified
workspace; a joint that stops following for 0.5 s holds the arm and freezes following;
Esc holds and exits (torque stays on). +/- change the speed limit live (5 % steps).
"""

from __future__ import annotations

import json
import time

import numpy as np

from so101_reach.real.controller import ReachController
from so101_reach.real.robot import RealPlant
from so101_reach.sim import so101
from so101_reach.sim.workspace import load_bank

LIMITS_PATH = so101.PROJECT_ROOT / "configs" / "real_workspace.json"


class SafeTargets:
  """Cursor point -> safe reach target (site box, min height, verified workspace)."""

  def __init__(self, min_z: float = 0.08):
    lim = json.loads(LIMITS_PATH.read_text()) if LIMITS_PATH.exists() else {}
    self.lo = np.array([lim.get("x_min", -1.0), lim.get("y_min", -1.0), max(lim.get("z_min", min_z), min_z)])
    self.hi = np.array([lim.get("x_max", 1.0), lim.get("y_max", 1.0), lim.get("z_max", 1.0)])
    pts = load_bank("eval").tcp_pos
    inside = np.all((pts >= self.lo) & (pts <= self.hi), axis=1)
    self.pts = pts[inside]
    if not len(self.pts):
      raise ValueError("No workspace targets fall inside the configured site bounds. Rebuild the workspace or review configs/real_workspace.json.")

  def __call__(self, raw: np.ndarray) -> tuple[np.ndarray, str]:
    box = np.clip(raw, self.lo, self.hi)
    d = np.linalg.norm(self.pts - box, axis=1)
    i = int(d.argmin())
    tgt = box if d[i] <= 0.01 else self.pts[i]
    moved = np.linalg.norm(tgt - raw)
    note = "" if moved < 0.005 else ("clipped by site limits" if np.linalg.norm(box - raw) > 0.005 else "snapped to reachable workspace")
    return tgt.astype(np.float64), note


class MouseFollowCore:
  """Control core, independent of the window (testable on a simulated bus)."""

  def __init__(
    self, plant: RealPlant, controller: ReachController, targets: SafeTargets, stuck_time: float = 0.5, source: str = "the mouse"
  ):
    self.source = source
    self.plant = plant
    self.ctrl = controller
    self.targets = targets
    self.stuck_steps = int(stuck_time / plant.dt)
    self.following = False
    self.message = f"FROZEN - click in the window or press F to follow {source}"
    q = plant.read_q()
    self.q = q
    self.tcp = controller.kin.tcp(q)
    self.target = self.tcp.copy()
    self.note = ""
    self._last_raw: np.ndarray | None = None
    self._stuck = 0
    controller.reset(q)

  def toggle(self) -> None:
    self.following = not self.following
    if self.following:
      self.message = f"FOLLOWING {self.source}"
    else:
      self.target = self.tcp.copy()
      self.message = f"FROZEN - click in the window or press F to follow {self.source}"

  def tick(self, cursor_hit: np.ndarray | None) -> None:
    """One 20 ms control step: read, choose target, policy, command."""
    q = self.plant.read_q()
    if self.plant.fixed:
      self.ctrl.kin.set_fixed(self.plant.fixed)
    if self.following and cursor_hit is not None:
      if self._last_raw is None or np.linalg.norm(cursor_hit - self._last_raw) > 0.002:
        self._last_raw = cursor_hit.copy()
        self.target, self.note = self.targets(cursor_hit)
    jt, tcp = self.ctrl.step(q, self.target, self.plant.dt)
    lag = np.abs(jt - q) >= self.ctrl.cfg.max_lead - 1e-3
    self._stuck = self._stuck + 1 if lag.any() else 0
    if self._stuck >= self.stuck_steps:
      joints = [n for n, b in zip(so101.ARM_JOINTS, lag, strict=True) if b]
      self.plant.hold()
      self.ctrl.reset(self.plant.read_q())
      self.following = False
      self.target = tcp.copy()
      self._stuck = 0
      self.message = f"SAFETY HOLD: {', '.join(joints)} blocked - clear the obstacle, then press F"
      return
    self.plant.command(jt)
    self.q, self.tcp = q, tcp


def run_window(core: MouseFollowCore, width: int = 1400, height: int = 900, plane_z: float | None = None) -> None:
  import glfw
  import mujoco

  from so101_reach.viz.mouse_follow import cursor_ray, intersect_plane_z

  m = so101.make_cpu_model()
  d = mujoco.MjData(m)
  qadr = {n: m.jnt_qposadr[m.joint(n).id] for n in so101.ALL_JOINTS}
  if not glfw.init():
    raise RuntimeError("GLFW init failed")
  glfw.window_hint(glfw.SAMPLES, 4)
  win = glfw.create_window(width, height, "SO-101 REAL ARM - follows the mouse (Esc: hold + quit)", None, None)
  glfw.make_context_current(win)
  glfw.swap_interval(0)
  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  cam.lookat[:] = (0.15, 0.0, 0.12)
  cam.distance, cam.azimuth, cam.elevation = 0.95, 180.0, -35.0
  opt = mujoco.MjvOption()
  scn = mujoco.MjvScene(m, maxgeom=2000)
  ctx = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150)
  state = {"plane_z": plane_z if plane_z is not None else float(core.tcp[2]), "buttons": set(), "last": (0.0, 0.0)}

  def on_button(w, button, action, mods):
    if action == glfw.PRESS:
      state["buttons"].add(button)
      if button == glfw.MOUSE_BUTTON_LEFT:
        core.toggle()
    else:
      state["buttons"].discard(button)

  def on_move(w, x, y):
    dx, dy = x - state["last"][0], y - state["last"][1]
    state["last"] = (x, y)
    h = max(glfw.get_window_size(w)[1], 1)
    if glfw.MOUSE_BUTTON_RIGHT in state["buttons"]:
      mujoco.mjv_moveCamera(m, mujoco.mjtMouse.mjMOUSE_ROTATE_V, dx / h, dy / h, scn, cam)
    elif glfw.MOUSE_BUTTON_MIDDLE in state["buttons"]:
      mujoco.mjv_moveCamera(m, mujoco.mjtMouse.mjMOUSE_MOVE_V, dx / h, dy / h, scn, cam)

  def on_scroll(w, xo, yo):
    if glfw.get_key(w, glfw.KEY_LEFT_SHIFT) == glfw.PRESS:
      mujoco.mjv_moveCamera(m, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * yo, scn, cam)
    else:
      state["plane_z"] = float(np.clip(state["plane_z"] + 0.005 * yo, 0.08, 0.45))

  def on_key(w, key, sc, action, mods):
    if action != glfw.PRESS:
      return
    if key == glfw.KEY_ESCAPE:
      glfw.set_window_should_close(w, True)
    elif key == glfw.KEY_F:
      core.toggle()
    elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD, glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
      step = 0.05 if key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD) else -0.05
      core.ctrl.cfg.speed_factor = float(np.clip(round(core.ctrl.cfg.speed_factor + step, 2), 0.05, 1.0))

  glfw.set_mouse_button_callback(win, on_button)
  glfw.set_cursor_pos_callback(win, on_move)
  glfw.set_scroll_callback(win, on_scroll)
  glfw.set_key_callback(win, on_key)

  def marker(gtype, size, pos, rgba):
    if scn.ngeom < scn.maxgeom:
      mujoco.mjv_initGeom(scn.geoms[scn.ngeom], gtype, np.asarray(size, float), np.asarray(pos, float), np.eye(3).flatten(), np.asarray(rgba, np.float32))
      scn.ngeom += 1

  deadline = time.perf_counter()
  try:
    while not glfw.window_should_close(win):
      fb_w, fb_h = glfw.get_framebuffer_size(win)
      vp = mujoco.MjrRect(0, 0, fb_w, fb_h)
      for n, v in zip(so101.ARM_JOINTS, core.q, strict=True):
        d.qpos[qadr[n]] = v
      for n, v in core.plant.fixed.items():
        d.qpos[qadr[n]] = v
      mujoco.mj_forward(m, d)
      mujoco.mjv_updateScene(m, d, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
      w, h = glfw.get_window_size(win)
      cx, cy = glfw.get_cursor_pos(win)
      hit = None
      if w and h:
        o, dr = cursor_ray(scn, cx / w, cy / h, fb_w / max(fb_h, 1))
        hit = intersect_plane_z(o, dr, state["plane_z"])
      core.tick(hit)
      err = np.linalg.norm(core.target - core.tcp) * 1e3
      if hit is not None:
        marker(mujoco.mjtGeom.mjGEOM_SPHERE, (0.006, 0, 0), hit, (1, 1, 1, 0.5))
      marker(mujoco.mjtGeom.mjGEOM_SPHERE, (0.012, 0, 0), core.target, (0.1, 0.9, 0.2, 0.7) if err < 10 else (1.0, 0.45, 0.1, 0.7))
      marker(mujoco.mjtGeom.mjGEOM_BOX, (0.15, 0.002, 0.15), (0.2, core.targets.hi[1] + 0.055, 0.15), (1.0, 0.1, 0.1, 0.12))
      mujoco.mjr_render(vp, scn, ctx)
      t, p = core.target, core.tcp
      left = "Mode\nSpeed\nTarget [m]\nTCP [m]\nError\nPlane height\nNote"
      right = (
        f"{core.message}\n{core.ctrl.cfg.speed_factor * 100:.0f} % (+/- keys)\n{t[0]:+.3f} {t[1]:+.3f} {t[2]:+.3f}\n{p[0]:+.3f} {p[1]:+.3f} {p[2]:+.3f}\n{err:.1f} mm\n"
        f"{state['plane_z']:.3f} m (scroll)\n{core.note or '-'}"
      )
      mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, vp, left, right, ctx)
      mujoco.mjr_overlay(
        mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, vp,
        "REAL ARM | left click / F: follow / freeze | +/-: speed | scroll: height | right drag: rotate | shift+scroll: zoom | Esc: hold + quit (red = wall keep-out)",
        "", ctx,
      )
      glfw.swap_buffers(win)
      glfw.poll_events()
      deadline += core.plant.dt
      delay = deadline - time.perf_counter()
      if delay > 0:
        time.sleep(delay)
      else:
        deadline = time.perf_counter()
  finally:
    core.plant.hold()
    glfw.terminate()
