"""Teach a grasp once by teleoperation, starting from the orange-hover pose.

States (one 50 Hz loop on the main thread; cameras run in their own threads):
  FOLLOW     follower hovers above the orange (reach policy); the leader mirrors the follower
             (low torque) so both arms match. F toggles following, R starts the countdown.
  COUNTDOWN  5 s: follower holds, leader keeps mirroring - take hold of the leader.
  TELEOP     leader released (limp); the follower copies it (LeRobot range mapping, all six
             joints incl. gripper); every step is recorded. R stops.
  DONE       demo saved; the follower holds its pose and grips gently (keeps the orange).

Saved (data/grasp_demos/grasp_<time>.npz + .json): follower / leader raw ticks, follower
goals, joint angles, TCP positions, the TCP path relative to the start (hover) point, gripper
opening, the orange position and hover settings at the start.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from so101_reach.real.feetech_bus import JOINT_ORDER, MotorCalib
from so101_reach.real.joint_map import JointMap
from so101_reach.real.mouse_real import MouseFollowCore
from so101_reach.real.orange_follow import OrangeTracker
from so101_reach.real.teleop import LeaderArm, RangeMap, rate_limit
from so101_reach.sim import so101

DEMO_DIR: Path = so101.PROJECT_ROOT / "data" / "grasp_demos"


class GraspTeachSession:
  def __init__(
    self,
    core: MouseFollowCore,
    tracker: OrangeTracker,
    leader: LeaderArm,
    fcal: dict[str, MotorCalib],
    lcal: dict[str, MotorCalib],
    countdown: float = 5.0,
    max_step: int = 40,
    gripper_torque: int = 350,
    hold_torque: int = 250,
    hold_squeeze: int = 30,
    method: str = "orange",
    obj: str = "orange",
  ):
    self.method, self.obj = method, obj
    self.core = core
    self.tr = tracker
    self.leader = leader
    self.fcal = fcal
    self.f2l, self.l2f = RangeMap(fcal, lcal), RangeMap(lcal, fcal)
    self.jmap = JointMap.load()
    self.countdown_s = countdown
    self.max_step = max_step  # follower goal change per step while teleoperating [ticks]
    self.gripper_torque = gripper_torque
    self.hold_torque = hold_torque
    self.hold_squeeze = hold_squeeze  # ticks the hold goal lies inside the contact position
    self.state = "FOLLOW"
    self.t_state = time.time()
    self.goal: dict[str, int] | None = None
    self.rec: dict[str, list] = {}
    self.meta: dict = {}
    self.saved: Path | None = None
    self.summary = ""

  @property
  def bus(self):
    return self.core.plant.bus

  def press_r(self) -> None:
    if self.state == "FOLLOW":
      self.state, self.t_state = "COUNTDOWN", time.time()
      if self.core.following:
        self.core.toggle()  # freeze: hold the hover pose
    elif self.state == "TELEOP":
      self._finish()

  def press_f(self) -> None:
    if self.state == "FOLLOW":
      self.core.toggle()

  def _clamp_follower(self, raw: dict[str, int]) -> dict[str, int]:
    return {n: int(np.clip(v, self.fcal[n].range_min, self.fcal[n].range_max)) for n, v in raw.items()}

  def _start_teleop(self) -> None:
    present = self.bus.sync_read("Present_Position")
    self.leader.release()
    self.bus.write("gripper", "Torque_Limit", self.gripper_torque)
    self.goal = self._clamp_follower(present)
    q0 = self._q(present)
    tcp0 = self._tcp(present)
    self.meta = {
      "created": datetime.now().isoformat(timespec="seconds"),
      "method": self.method,
      "object": self.obj,
      "orange_center_base": None if self.tr.center is None else self.tr.center.tolist(),
      "orange_radius": float(self.tr.radius),
      "orange_seen_by": self.tr.seen_by,
      "hover_m": float(self.tr.hover),
      "start_tcp_base": tcp0.tolist(),
      "start_joint_rad": q0.tolist(),
      "start_follower_raw": [present[n] for n in JOINT_ORDER],
      "joint_order": list(JOINT_ORDER),
      "mapping": "LeRobot range mapping (use_degrees=False), leader -> follower",
      "rate_hz": 1.0 / self.core.plant.dt,
    }
    self.rec = {k: [] for k in ("t", "follower_raw", "follower_goal", "leader_raw", "q", "tcp")}
    self.state, self.t_state = "TELEOP", time.time()

  def _q(self, raw: dict[str, int]) -> np.ndarray:
    return np.array([self.jmap.to_rad(n, raw[n]) for n in JOINT_ORDER])

  def _tcp(self, raw: dict[str, int]) -> np.ndarray:
    q = self._q(raw)
    kin = self.core.ctrl.kin
    kin.set_fixed({"wrist_roll": q[4], "gripper": q[5]})
    return kin.tcp(q[:4])

  def step(self, cursor_target: np.ndarray | None) -> None:
    if self.state in ("FOLLOW", "COUNTDOWN"):
      self.core.tick(cursor_target if self.core.following else None)
      if self.core.plant.last_raw:
        self.leader.mirror(self.f2l(self.core.plant.last_raw))
      if self.state == "COUNTDOWN" and time.time() - self.t_state >= self.countdown_s:
        self._start_teleop()
    elif self.state == "TELEOP":
      lraw = self.leader.read()
      present = self.bus.sync_read("Present_Position")
      self.goal = self._clamp_follower(rate_limit(self.l2f(lraw), self.goal, self.max_step))
      self.bus.sync_write("Goal_Position", self.goal)
      r = self.rec
      r["t"].append(time.time() - self.t_state)
      r["follower_raw"].append([present[n] for n in JOINT_ORDER])
      r["follower_goal"].append([self.goal[n] for n in JOINT_ORDER])
      r["leader_raw"].append([lraw[n] for n in JOINT_ORDER])
      r["q"].append(self._q(present))
      r["tcp"].append(self._tcp(present))

  def _finish(self) -> None:
    present = self.bus.sync_read("Present_Position")
    hold = dict(present)
    # Gentle grip: a goal just inside the contact position at reduced torque. Holding the
    # leader's fully-closed goal at high torque stalls the STS3215 (overload protection).
    closing = self.goal["gripper"] < present["gripper"]
    hold["gripper"] = present["gripper"] - self.hold_squeeze if closing else present["gripper"]
    self.bus.write("gripper", "Torque_Limit", self.hold_torque)
    self.bus.sync_write("Goal_Position", self._clamp_follower(hold))
    self.state, self.t_state = "DONE", time.time()
    self.saved = self.save()

  def save(self) -> Path:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    arr = {k: np.array(v) for k, v in self.rec.items()}
    tcp = arr["tcp"]
    arr["tcp_rel"] = tcp - np.array(self.meta["start_tcp_base"])
    g = arr["follower_raw"][:, 5]
    gc = self.fcal["gripper"]
    arr["gripper_open"] = (g - gc.range_min) / (gc.range_max - gc.range_min)  # 0 closed .. 1 fully open
    path = DEMO_DIR / f"grasp_{stamp}.npz"
    np.savez(path, **arr)
    rel = arr["tcp_rel"]
    self.meta.update(
      duration_s=float(arr["t"][-1]) if len(arr["t"]) else 0.0,
      samples=int(len(arr["t"])),
      lowest_tcp_rel_z=float(rel[:, 2].min()) if len(rel) else 0.0,
      final_tcp_rel=rel[-1].tolist() if len(rel) else [0, 0, 0],
      gripper_open_range=[float(arr["gripper_open"].min()), float(arr["gripper_open"].max())] if len(g) else [0, 0],
    )
    path.with_suffix(".json").write_text(json.dumps(self.meta, indent=2) + "\n")
    m = self.meta
    self.summary = (
      f"{m['samples']} samples, {m['duration_s']:.1f} s; TCP went down {-m['lowest_tcp_rel_z'] * 100:.1f} cm, "
      f"ended {m['final_tcp_rel'][2] * 100:+.1f} cm vs start; gripper opening {m['gripper_open_range'][0]:.2f}-{m['gripper_open_range'][1]:.2f}"
    )
    return path

  def status(self) -> str:
    if self.state == "FOLLOW":
      return "TEACH: R = start teleop (5 s countdown)" + ("" if self.core.following else " | F/click = follow the orange")
    if self.state == "COUNTDOWN":
      return f"TAKE THE LEADER ARM - teleop starts in {max(0.0, self.countdown_s - (time.time() - self.t_state)):.1f} s"
    if self.state == "TELEOP":
      return f"RECORDING {time.time() - self.t_state:5.1f} s ({len(self.rec['t'])} samples) - open, go down, grasp, lift; R = stop"
    return f"SAVED {self.saved.name if self.saved else ''}: {self.summary} | Esc = quit (arm holds)"
