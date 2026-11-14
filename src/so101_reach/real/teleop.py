"""Leader arm (SO-101 leader) <-> follower mapping, mirroring and teleoperation.

Mapping: LeRobot's SO-101 default (`use_degrees=False`, as used by So-101_Arm/teleop.sh):
every joint, gripper included, is mapped proportionally between the two arms' calibrated
ranges (`range_min`..`range_max` of the LeRobot calibration files), so teleoperation feels
the same as `lerobot_teleoperate`.

Mirroring: the leader is position-controlled (low torque, rate-limited goals) to copy the
follower, so the two arms match before the user takes over.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from so101_reach.real.feetech_bus import JOINT_ORDER, FeetechBus, MotorCalib, load_calibration

from so101_reach.settings import LEADER_PORT, LEADER_CALIB


class RangeMap:
  """Raw ticks of one arm -> raw ticks of the other (proportional between calibrated ranges)."""

  def __init__(self, src: dict[str, MotorCalib], dst: dict[str, MotorCalib]):
    for c in (*src.values(), *dst.values()):
      assert c.drive_mode == 0, "drive_mode != 0 not supported"
    self.src, self.dst = src, dst

  def __call__(self, raw: dict[str, int]) -> dict[str, int]:
    out = {}
    for n in JOINT_ORDER:
      s, d = self.src[n], self.dst[n]
      f = (np.clip(raw[n], s.range_min, s.range_max) - s.range_min) / (s.range_max - s.range_min)
      out[n] = int(round(d.range_min + f * (d.range_max - d.range_min)))
    return out


def rate_limit(goal: dict[str, int], prev: dict[str, int], max_step: int) -> dict[str, int]:
  return {n: prev[n] + int(np.clip(goal[n] - prev[n], -max_step, max_step)) for n in goal}


class LeaderArm:
  def __init__(self, bus: FeetechBus, torque_limit: int = 300, max_step: int = 15):
    self.bus = bus
    self.torque_limit = torque_limit
    self.max_step = max_step  # ticks per 20 ms step (~66 deg/s) while mirroring
    self.goal: dict[str, int] | None = None
    self.engaged = False

  def read(self) -> dict[str, int]:
    return self.bus.sync_read("Present_Position")

  def engage(self) -> None:
    """Torque on, holding the present pose (no jump)."""
    pos = self.read()
    self.bus.sync_write("Goal_Position", pos)
    for n in JOINT_ORDER:
      self.bus.write(n, "Torque_Limit", self.torque_limit)
    self.bus.sync_write("Torque_Enable", dict.fromkeys(JOINT_ORDER, 1))
    self.goal = pos
    self.engaged = True

  def mirror(self, goal: dict[str, int]) -> None:
    """Move toward `goal` (leader raw ticks), rate-limited."""
    if not self.engaged:
      return
    self.goal = rate_limit(goal, self.goal, self.max_step)
    self.bus.sync_write("Goal_Position", self.goal)

  def release(self) -> None:
    self.bus.sync_write("Torque_Enable", dict.fromkeys(JOINT_ORDER, 0))
    self.engaged = False


def open_leader(port: str = LEADER_PORT) -> tuple[FeetechBus, dict[str, MotorCalib]]:
  calib = load_calibration(LEADER_CALIB)
  return FeetechBus(port, calib=calib), calib
