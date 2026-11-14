"""Real SO-101 follower as a controller plant, with safety handling.

Engage sequence (no jump): read Present_Position -> Goal_Position := Present_Position ->
Torque_Limit := limit -> Torque_Enable := 1. Every arm goal is clamped to the calibrated
servo range. `hold()` re-targets every joint to its present position (arm stays up);
`release()` disables torque (the arm falls - only call deliberately).

`SimServoBus` emulates the Feetech bus on a CPU MuJoCo plant so the full live code path
(registers, tick conversion, watchdog) can be tested without hardware.
"""

from __future__ import annotations

import time

import numpy as np

from so101_reach.real.controller import SimPlant
from so101_reach.real.feetech_bus import JOINT_ORDER, MotorCalib
from so101_reach.real.joint_map import JointMap
from so101_reach.sim import so101


class SafetyStop(RuntimeError):
  """Raised by run_live; `record` holds the trajectory up to the stop (for logging)."""

  record: dict[str, np.ndarray] | None = None


class RealPlant:
  dt = 0.02

  def __init__(self, bus, jmap: JointMap, calib: dict[str, MotorCalib], torque_limit: int = 400, max_read_failures: int = 3):
    self.bus = bus
    self.jmap = jmap
    self.calib = calib
    self.torque_limit = torque_limit
    self.max_read_failures = max_read_failures
    self._deadline = None
    self.overruns = 0
    self.engaged = False
    self.fixed: dict[str, float] = {}
    self.last_raw: dict[str, int] = {}

  def _present(self) -> dict[str, int]:
    for attempt in range(self.max_read_failures):
      try:
        return self.bus.sync_read("Present_Position")
      except IOError:
        if attempt == self.max_read_failures - 1:
          raise SafetyStop("servo read failed repeatedly") from None
    raise AssertionError

  def engage(self, keep_gripper: bool = False) -> None:
    """Torque on holding the present pose. keep_gripper: leave the gripper's goal and torque
    limit untouched (e.g. while it holds an object)."""
    pos = self._present()
    names = [n for n in JOINT_ORDER if not (keep_gripper and n == "gripper")]
    self.bus.sync_write("Goal_Position", {n: self._clamp(n, pos[n]) for n in names})
    for n in names:
      # The closed gripper rests just below its calibrated minimum; a low limit stops it
      # squeezing (and heating) while it holds.
      lim = min(self.torque_limit, 150) if n == "gripper" else self.torque_limit
      self.bus.write(n, "Torque_Limit", lim)
    self.bus.sync_write("Torque_Enable", {n: 1 for n in names})
    self.engaged = True
    self._deadline = time.perf_counter()

  def _clamp(self, name: str, ticks: int) -> int:
    c = self.calib[name]
    return int(np.clip(ticks, c.range_min, c.range_max))

  def read_q(self) -> np.ndarray:
    pos = self._present()
    self.last_raw = pos
    self.fixed = {n: self.jmap.to_rad(n, pos[n]) for n in so101.FIXED_JOINTS}
    return np.array([self.jmap.to_rad(n, pos[n]) for n in so101.ARM_JOINTS])

  def command(self, joint_target: np.ndarray) -> None:
    goal = {n: self._clamp(n, self.jmap.to_ticks(n, v)) for n, v in zip(so101.ARM_JOINTS, joint_target, strict=True)}
    self.bus.sync_write("Goal_Position", goal)

  def wait(self) -> None:
    self._deadline += self.dt
    delay = self._deadline - time.perf_counter()
    if delay > 0:
      time.sleep(delay)
    else:
      self.overruns += 1
      self._deadline = time.perf_counter()

  def hold(self) -> None:
    pos = self.bus.sync_read("Present_Position")
    self.bus.sync_write("Goal_Position", {n: self._clamp(n, pos[n]) for n in JOINT_ORDER})

  def release(self) -> None:
    self.bus.sync_write("Torque_Enable", {n: 0 for n in JOINT_ORDER})
    self.engaged = False


class SimServoBus:
  """Feetech-bus look-alike backed by SimPlant (ticks in, ticks out, torque gating)."""

  def __init__(self, jmap: JointMap, q0: np.ndarray):
    self.jmap = jmap
    self.plant = SimPlant(q0, quantize=False)
    self.torque = dict.fromkeys(JOINT_ORDER, 0)
    self.regs: dict[tuple[str, str], int] = {}

  def sync_read(self, reg: str, names=JOINT_ORDER) -> dict[str, int]:
    assert reg in ("Present_Position", "Torque_Enable")
    if reg == "Torque_Enable":
      return {n: self.torque[n] for n in names}
    q = self.plant.data.qpos
    m = self.plant.model
    return {n: self.jmap.to_ticks(n, q[m.jnt_qposadr[m.joint(n).id]]) for n in names}

  def sync_write(self, reg: str, values: dict[str, int]) -> None:
    if reg == "Torque_Enable":
      self.torque.update(values)
    elif reg == "Goal_Position":
      m = self.plant.model
      for n, t in values.items():
        self.plant.data.ctrl[m.actuator(n).id] = self.jmap.to_rad(n, t)
    for n, v in values.items():
      self.regs[(n, reg)] = v

  def write(self, name: str, reg: str, value: int) -> None:
    self.regs[(name, reg)] = value

  def step_physics(self) -> None:
    self.plant.wait()


class SimRealPlant(RealPlant):
  """RealPlant on a SimServoBus: physics advances in wait() instead of sleeping."""

  def wait(self) -> None:
    self.bus.step_physics()


def run_live(
  plant: RealPlant,
  controller,
  target: np.ndarray,
  duration: float,
  settle_tol: float = 0.005,
  settle_time: float = 0.5,
  stuck_time: float = 0.5,
  diverge_margin: float = 0.02,
) -> dict[str, np.ndarray]:
  """Closed-loop reach on an engaged plant. Stops when settled or after `duration`.

  Raises SafetyStop if the commanded target is pinned at the max-lead clamp for
  `stuck_time` (arm blocked / not following), if the TCP error grows more than
  `diverge_margin` above its initial value (e.g. a mirrored joint), or if servo reads fail.
  """
  q = plant.read_q()
  controller.reset(q)
  max_lead = controller.cfg.max_lead
  rec: dict[str, list] = {"t": [], "q": [], "joint_target": [], "tcp": [], "err": []}
  stuck = settled = 0
  err0 = None
  t0 = time.perf_counter()
  for k in range(int(duration / plant.dt)):
    q = plant.read_q()
    if getattr(plant, "fixed", None):
      controller.kin.set_fixed(plant.fixed)
    jt, tcp = controller.step(q, target, plant.dt)
    err = float(np.linalg.norm(target - tcp))
    err0 = err if err0 is None else err0
    lagging = np.abs(jt - q) >= max_lead - 1e-3
    stuck = stuck + 1 if np.any(lagging) else 0
    reason = None
    if err > err0 + diverge_margin:
      reason = f"TCP error grew from {err0 * 1e3:.0f} to {err * 1e3:.0f} mm (mirrored joint / wrong map?)"
    elif stuck * plant.dt >= stuck_time:
      joints = [n for n, lag in zip(so101.ARM_JOINTS, lagging, strict=True) if lag]
      reason = f"{', '.join(joints)} not following its command for {stuck_time:.1f} s (blocked?) at t={k * plant.dt:.2f} s"
    if reason is not None:
      for key, v in (("t", time.perf_counter() - t0), ("q", q), ("joint_target", jt), ("tcp", tcp), ("err", err)):
        rec[key].append(v)
      stop = SafetyStop(reason)
      stop.record = {key: np.array(v) for key, v in rec.items()}
      raise stop
    plant.command(jt)
    for key, v in (("t", time.perf_counter() - t0), ("q", q), ("joint_target", jt), ("tcp", tcp), ("err", err)):
      rec[key].append(v)
    settled = settled + 1 if err < settle_tol else 0
    if settled * plant.dt >= settle_time:
      break
    plant.wait()
  return {k: np.array(v) for k, v in rec.items()}
