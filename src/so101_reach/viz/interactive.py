"""Interactive coordinate-commanded reaching viewers.

* ``ReachViserViewer`` (default): browser GUI (http://localhost:8080) built on MjLab's
  Viser viewer. Panel "Reach target": x/y/z inputs + Go, Random target, Home, auto-random
  sequence, and a live status block (target, TCP, error [mm], speed, joints, elapsed time,
  MOVING / REACHED / FAILED, workspace feasibility).
* ``ReachNativeViewer``: MuJoCo desktop window with the same status as an overlay;
  targets are typed into the terminal ("0.20 0.10 0.15", "r" = random, "h" = home).

GUI/terminal events are queued with ``request_action`` and executed on the simulation
thread, so commands never race with physics stepping.
"""

from __future__ import annotations

import threading
import time

import mujoco
import numpy as np

from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.base import ViewerAction

from so101_reach.viz.session import ReachSession


class _ReachMixin:
  session: ReachSession
  auto_random: bool = False
  _auto_wait_until: float | None = None

  def _handle_reach_action(self, payload) -> bool:
    if not isinstance(payload, dict) or "reach" not in payload:
      return False
    kind = payload["reach"]
    if kind == "go":
      msg = self.session.command(payload["xyz"])
      print(f"[reach] target {np.round(payload['xyz'], 3)} m -> {msg}")
    elif kind == "random":
      xyz = self.session.random_target()
      print(f"[reach] random target {np.round(xyz, 3)} m")
    elif kind == "home":
      self.session.home()
    elif kind == "auto":
      self.auto_random = bool(payload["on"])
    return True

  def _auto_sequence(self) -> None:
    if not self.auto_random:
      return
    st = self.session.status()
    if st.state.startswith(("REACHED", "FAILED")):
      now = time.perf_counter()
      if self._auto_wait_until is None:
        self._auto_wait_until = now + 0.8
      elif now >= self._auto_wait_until:
        self._auto_wait_until = None
        self.session.random_target()


class ReachViserViewer(_ReachMixin, ViserPlayViewer):
  def __init__(self, env, policy, session: ReachSession, **kw):
    super().__init__(env, policy, **kw)
    self.session = session

  def setup(self) -> None:
    super().setup()
    gui = self._server.gui
    s0 = self.session.status()
    with gui.add_folder("Reach target", order=-10):
      x = gui.add_number("x [m]", initial_value=float(np.round(s0.target[0], 3)), step=0.005)
      y = gui.add_number("y [m]", initial_value=float(np.round(s0.target[1], 3)), step=0.005)
      z = gui.add_number("z [m]", initial_value=float(np.round(s0.target[2], 3)), step=0.005)
      go = gui.add_button("Go to target")
      rnd = gui.add_button("Random target")
      home = gui.add_button("Home")
      auto = gui.add_checkbox("Auto random sequence", initial_value=False)
      self._status_md = gui.add_markdown("")

    @go.on_click
    def _(_e) -> None:
      self.request_action("CUSTOM", {"reach": "go", "xyz": [x.value, y.value, z.value]})

    @rnd.on_click
    def _(_e) -> None:
      self.request_action("CUSTOM", {"reach": "random"})

    @home.on_click
    def _(_e) -> None:
      self.request_action("CUSTOM", {"reach": "home"})

    @auto.on_update
    def _(_e) -> None:
      self.request_action("CUSTOM", {"reach": "auto", "on": auto.value})

    self._xyz_inputs = (x, y, z)

  def _handle_custom_action(self, action: ViewerAction, payload) -> bool:
    if self._handle_reach_action(payload):
      if payload["reach"] in ("random", "home"):
        t = self.session.status().target
        for h, v in zip(self._xyz_inputs, t, strict=True):
          h.value = float(np.round(v, 3))
      return True
    return super()._handle_custom_action(action, payload)

  def sync_env_to_viewer(self) -> None:
    super().sync_env_to_viewer()
    self._auto_sequence()
    if self._counter % 6 == 0:
      st = self.session.status()
      color = {"REACHED": "#27ae60", "MOVING": "#2980b9"}.get(st.state.split()[0], "#c0392b")
      rows = "\n".join(f"| {k} | {v} |" for k, v in st.lines()[1:])
      head = f"<span style='color:{color};font-weight:bold'>{st.lines()[0][1]}</span>"
      self._status_md.content = f"{head}\n\n| | |\n|---|---|\n{rows}"


class ReachNativeViewer(_ReachMixin, NativeMujocoViewer):
  def __init__(self, env, policy, session: ReachSession, **kw):
    super().__init__(env, policy, **kw)
    self.session = session
    self._stdin_thread = threading.Thread(target=self._stdin_loop, daemon=True)

  def setup(self) -> None:
    super().setup()
    self._stdin_thread.start()

  def _stdin_loop(self) -> None:
    print("\nEnter a target as 'x y z' in metres (base frame), 'r' random, 'a' toggle auto-random, 'h' home, 'q' quit.")
    while True:
      try:
        line = input("target> ").strip().lower()
      except EOFError:
        return
      if not line:
        continue
      if line == "q":
        self._interrupted = True
        return
      if line == "r":
        self.request_action("CUSTOM", {"reach": "random"})
      elif line == "h":
        self.request_action("CUSTOM", {"reach": "home"})
      elif line == "a":
        self.request_action("CUSTOM", {"reach": "auto", "on": not self.auto_random})
      else:
        try:
          xyz = [float(v) for v in line.replace(",", " ").replace("=", " ").split() if _is_float(v)]
          if len(xyz) != 3:
            raise ValueError
          self.request_action("CUSTOM", {"reach": "go", "xyz": xyz})
        except ValueError:
          print("  expected three numbers, e.g. 0.20 0.10 0.15  or  x=0.20 y=0.10 z=0.15")

  def _handle_custom_action(self, action: ViewerAction, payload) -> bool:
    if self._handle_reach_action(payload):
      return True
    return super()._handle_custom_action(action, payload)

  def _set_status_overlay(self, viewer: mujoco.viewer.Handle) -> None:
    self._auto_sequence()
    lines = self.session.status().lines()
    viewer.set_texts(
      (
        mujoco.mjtFontScale.mjFONTSCALE_150.value,
        mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
        "\n".join(k for k, _ in lines),
        "\n".join(v for _, v in lines),
      )
    )


def _is_float(s: str) -> bool:
  try:
    float(s)
    return True
  except ValueError:
    return False
