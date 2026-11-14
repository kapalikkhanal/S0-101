"""MuJoCo window in which the reach target follows the mouse pointer.

The cursor is cast as a ray from the camera and intersected with a horizontal plane at a
user-controlled height; the hit point is projected onto the verified workspace and fed to
the controller as the target every frame. Physics and the policy run in the same MjLab
evaluation environment used for benchmarking (MuJoCo Warp, 50 Hz policy); the window shows a
CPU mirror of its state.

Controls:
  move mouse          target follows the cursor (on the height plane)
  scroll              raise / lower the height plane (5 mm per notch)
  shift + scroll      zoom
  right drag          rotate camera
  middle drag         pan camera
  left click / F      freeze / resume following
  R                   random feasible target (freezes following)
  H                   home target (freezes following)
  space               pause / resume simulation
  Esc                 quit
"""

from __future__ import annotations

import time

import glfw
import mujoco
import numpy as np
import torch

from so101_reach.viz.session import ReachSession


def cursor_ray(scn: mujoco.MjvScene, x: float, y: float, aspect: float) -> tuple[np.ndarray, np.ndarray]:
  """World-space ray through normalised window coordinates (x right, y down, both in [0, 1])."""
  c0, c1 = scn.camera[0], scn.camera[1]
  pos = 0.5 * (np.array(c0.pos) + np.array(c1.pos))
  fwd = 0.5 * (np.array(c0.forward) + np.array(c1.forward))
  up = 0.5 * (np.array(c0.up) + np.array(c1.up))
  fwd /= np.linalg.norm(fwd)
  up /= np.linalg.norm(up)
  right = np.cross(fwd, up)
  near, bottom, top = c0.frustum_near, c0.frustum_bottom, c0.frustum_top
  half_w = c0.frustum_width if c0.frustum_width > 0 else 0.5 * (top - bottom) * aspect
  px = c0.frustum_center + (2.0 * x - 1.0) * half_w
  py = bottom + (1.0 - y) * (top - bottom)
  d = fwd * near + right * px + up * py
  return pos, d / np.linalg.norm(d)


def intersect_plane_z(origin: np.ndarray, direction: np.ndarray, z: float) -> np.ndarray | None:
  if abs(direction[2]) < 1e-6:
    return None
  t = (z - origin[2]) / direction[2]
  return origin + t * direction if t > 0 else None


class MouseFollowApp:
  def __init__(self, session: ReachSession, width: int = 1400, height: int = 900, plane_z: float = 0.12):
    self.s = session
    self.env = session.env
    self.m = self.env.sim.mj_model
    self.d = mujoco.MjData(self.m)
    self.plane_z = plane_z
    self.follow = True
    self.paused = False
    self.cursor_hit: np.ndarray | None = None
    self.target = session.status().target.copy()
    self._buttons = {glfw.MOUSE_BUTTON_RIGHT: False, glfw.MOUSE_BUTTON_MIDDLE: False}
    self._last = (0.0, 0.0)
    self._shift = False

    if not glfw.init():
      raise RuntimeError("GLFW init failed (is a display available?)")
    glfw.window_hint(glfw.SAMPLES, 4)
    self.win = glfw.create_window(width, height, "SO-101 reach - target follows mouse", None, None)
    if not self.win:
      glfw.terminate()
      raise RuntimeError("Could not create window")
    glfw.make_context_current(self.win)
    glfw.swap_interval(1)

    self.cam = mujoco.MjvCamera()
    self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    self.cam.lookat[:] = (0.15, 0.0, 0.12)
    self.cam.distance, self.cam.azimuth, self.cam.elevation = 0.95, 145.0, -28.0
    self.opt = mujoco.MjvOption()
    self.scn = mujoco.MjvScene(self.m, maxgeom=2000)
    self.ctx = mujoco.MjrContext(self.m, mujoco.mjtFontScale.mjFONTSCALE_150)

    glfw.set_cursor_pos_callback(self.win, self._on_move)
    glfw.set_mouse_button_callback(self.win, self._on_button)
    glfw.set_scroll_callback(self.win, self._on_scroll)
    glfw.set_key_callback(self.win, self._on_key)

  # Input.

  def _on_move(self, win, xpos, ypos) -> None:
    dx, dy = xpos - self._last[0], ypos - self._last[1]
    self._last = (xpos, ypos)
    w, h = glfw.get_window_size(win)
    if self._buttons[glfw.MOUSE_BUTTON_RIGHT]:
      mujoco.mjv_moveCamera(self.m, mujoco.mjtMouse.mjMOUSE_ROTATE_V, dx / h, dy / h, self.scn, self.cam)
    elif self._buttons[glfw.MOUSE_BUTTON_MIDDLE]:
      mujoco.mjv_moveCamera(self.m, mujoco.mjtMouse.mjMOUSE_MOVE_V, dx / h, dy / h, self.scn, self.cam)

  def _on_button(self, win, button, action, mods) -> None:
    if button in self._buttons:
      self._buttons[button] = action == glfw.PRESS
    elif button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
      self.follow = not self.follow

  def _on_scroll(self, win, xoff, yoff) -> None:
    shift = glfw.get_key(win, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or glfw.get_key(win, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    if shift:
      mujoco.mjv_moveCamera(self.m, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * yoff, self.scn, self.cam)
    else:
      self.plane_z = float(np.clip(self.plane_z + 0.005 * yoff, 0.02, 0.55))

  def _on_key(self, win, key, scancode, action, mods) -> None:
    if action != glfw.PRESS:
      return
    if key == glfw.KEY_ESCAPE:
      glfw.set_window_should_close(win, True)
    elif key == glfw.KEY_F:
      self.follow = not self.follow
    elif key == glfw.KEY_SPACE:
      self.paused = not self.paused
    elif key == glfw.KEY_R:
      self.follow = False
      self.target = self.s.random_target()
    elif key == glfw.KEY_H:
      self.follow = False
      self.s.home()
      self.target = self.s.status().target.copy()

  # Loop.

  def _update_target_from_cursor(self, aspect: float) -> None:
    w, h = glfw.get_window_size(self.win)
    cx, cy = glfw.get_cursor_pos(self.win)
    if w == 0 or h == 0:
      return
    origin, direction = cursor_ray(self.scn, cx / w, cy / h, aspect)
    self.cursor_hit = intersect_plane_z(origin, direction, self.plane_z)
    if self.follow and self.cursor_hit is not None:
      self.target = self.s.track(self.cursor_hit)

  def _sync_mirror(self) -> None:
    sd = self.env.sim.data
    self.d.qpos[:] = sd.qpos[0].cpu().numpy()
    self.d.qvel[:] = sd.qvel[0].cpu().numpy()
    if self.m.nmocap:
      self.d.mocap_pos[:] = sd.mocap_pos[0].cpu().numpy()
      self.d.mocap_quat[:] = sd.mocap_quat[0].cpu().numpy()
    mujoco.mj_forward(self.m, self.d)

  def _add_marker(self, gtype, size, pos, rgba) -> None:
    if self.scn.ngeom >= self.scn.maxgeom:
      return
    g = self.scn.geoms[self.scn.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.asarray(size, float), np.asarray(pos, float), np.eye(3).flatten(), np.asarray(rgba, np.float32))
    self.scn.ngeom += 1

  def _add_line(self, a, b, rgba, width=2.0) -> None:
    if self.scn.ngeom >= self.scn.maxgeom:
      return
    g = self.scn.geoms[self.scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.eye(3).flatten(), np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, width, np.asarray(a, float), np.asarray(b, float))
    self.scn.ngeom += 1

  def run(self) -> None:
    step_dt = self.env.step_dt
    budget, last = 0.0, time.perf_counter()
    while not glfw.window_should_close(self.win):
      now = time.perf_counter()
      budget = min(budget + (now - last), 5 * step_dt)
      last = now
      fb_w, fb_h = glfw.get_framebuffer_size(self.win)
      viewport = mujoco.MjrRect(0, 0, fb_w, fb_h)
      aspect = fb_w / max(fb_h, 1)

      self._sync_mirror()
      mujoco.mjv_updateScene(self.m, self.d, self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scn)
      self._update_target_from_cursor(aspect)

      if not self.paused:
        while budget >= step_dt:
          with torch.no_grad():
            self.env.step(self.s.policy(self.env.get_observations()))
          budget -= step_dt
      else:
        budget = 0.0

      st = self.s.status()
      ok = st.error_mm < self.s.cmd.stage.success_tol * 1e3
      if self.cursor_hit is not None:
        c = self.cursor_hit
        self._add_marker(mujoco.mjtGeom.mjGEOM_SPHERE, (0.006, 0, 0), c, (1, 1, 1, 0.5))
        self._add_marker(mujoco.mjtGeom.mjGEOM_CYLINDER, (0.04, 0.0005, 0), c, (1, 1, 1, 0.15))
        self._add_line((c[0], c[1], 0.0), c, (1, 1, 1, 0.6))
      t = st.target
      self._add_marker(mujoco.mjtGeom.mjGEOM_SPHERE, (0.012, 0, 0), t, (0.1, 0.9, 0.2, 0.7) if ok else (1.0, 0.45, 0.1, 0.7))
      self._add_line((t[0], t[1], 0.0), t, (1.0, 0.6, 0.2, 0.8))
      self._add_marker(mujoco.mjtGeom.mjGEOM_SPHERE, (0.006, 0, 0), st.tcp, (0.1, 0.4, 1.0, 0.9))

      mujoco.mjr_render(viewport, self.scn, self.ctx)
      lines = st.lines() + [
        ("Plane height", f"{self.plane_z:.3f} m (scroll)"),
        ("Mode", ("FOLLOWING mouse" if self.follow else "FROZEN (click / F)") + (" | PAUSED" if self.paused else "")),
      ]
      mujoco.mjr_overlay(
        mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport,
        "\n".join(k for k, _ in lines), "\n".join(v for _, v in lines), self.ctx,
      )
      mujoco.mjr_overlay(
        mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, viewport,
        "mouse: target | scroll: height | shift+scroll: zoom | right/middle drag: camera | click/F: freeze | R random | H home | space pause | Esc quit",
        "", self.ctx,
      )
      glfw.swap_buffers(self.win)
      glfw.poll_events()
    glfw.terminate()
