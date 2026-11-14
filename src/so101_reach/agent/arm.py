"""The agent's arm tools: what the LLM can ask the arm to do, with the safety rules in code.

Units for the agent: cm and degrees; x forward, y left; heights `h` above the table.
All positions refer to the GRIP POINT: midway between the two jaw tips, 1.5 cm in from the
tips - where an object sits when the jaws close on it (it moves with the opening).
  move_to(x, y, h, tilt, jaw_yaw)  long moves (> 6 cm): straight up to the travel height (~9 cm),
                                   RL reach policy across, then straight in; short: IK straight line
  move_by(dx, dy, dz)              IK straight line, orientation kept
  set_orientation(tilt, jaw_yaw)   IK, grip point kept in place
  gripper(open_percent, force)     closes until contact (jaws stop short), then holds with a
                                   force-dependent squeeze; or opens to a percentage
  home()
tilt: gripper angle from straight down, + = tips leaning away from the robot (radial plane).
jaw_yaw: direction (deg in the table plane, 0 = +x, 90 = +y) along which the jaws close.

Safety (refused before moving, or the arm stops and holds): the jaw tips never below
`floor_cm` above the table, xy inside the site limits (configs/real_workspace.json, e.g. the
wall), IK must reach the pose (<= 3 mm), and every move stops if an arm joint lags its
command (blocked by contact) for 0.3 s.

Gravity: the STS3215 position loop is proportional only (identified kp ~10 N.m/rad), so the
arm sags under its own weight - up to ~5 cm at full reach. Every joint plan gets a gravity
feedforward (MuJoCo gravity torque / kp added to the commanded angles), and after each move
the measured grip-point error is corrected (up to 3 re-aims).
"""

from __future__ import annotations

import time

import mujoco
import numpy as np
from scipy.optimize import least_squares

from so101_reach.real.feetech_bus import JOINT_ORDER
from so101_reach.real.grasp_replay import ArmKinematics, PlanPlayer, PolicyMove, ReplayPlan, _rz, joint_plan
from so101_reach.real.mouse_real import MouseFollowCore
from so101_reach.sim import so101

# Torque_Limit and squeeze (ticks past the contact position) per grip force. Kept moderate: a
# large squeeze at a high limit stalls the STS3215 into overload protection over a long hold.
# Parking pose that hides ~0.5 % of the table from the head camera (home: ~31 %): turned to the
# robot's right, tip at (6, -24, 18) cm. Found by projecting IK poses into the calibrated head view,
# keeping those without self-contact, every link >= 5 cm above the table and clear of the wall.
CLEAR_VIEW_DEG = {"shoulder_pan": 87.0, "shoulder_lift": -43.0, "elbow_flex": 28.0, "wrist_flex": 45.0}
FORCES = {"gentle": (200, 12), "medium": (300, 20), "firm": (400, 30)}
# Head-camera observation pose for the 2026-11-07 layout: folded left of the base, silhouette clear
# of the cube and box in the head view, D405 >= 7.7 cm from the wall (pan -20 let it reach the wall).
OBSERVE_DEG = {"shoulder_pan": -12.0, "shoulder_lift": -60.0, "elbow_flex": 60.0, "wrist_flex": 70.0}
GRIP_IN_FROM_TIPS = 0.015


class _ArmPlayer(PlanPlayer):
  """PlanPlayer whose stop / end hold only touches the arm joints: the gripper keeps its goal
  (a held object stays squeezed).

  The approach blend starts from the servos' present *goal*, not their measured position:
  P-only servos sit below a gravity-compensated goal, so resetting the goal to the measured
  pose drops the feedforward and makes the arm sag at the start of every segment."""

  MAX_GOAL_LEAD_TICKS = 120  # ~10 deg; a larger goal/position gap is not a plain sag offset

  def __init__(self, bus, plan, jmap, fcal, **kw):
    super().__init__(bus, plan, jmap, fcal, **kw)
    try:
      goal = bus.sync_read("Goal_Position")
    except (OSError, AssertionError, KeyError):
      return
    for n in JOINT_ORDER[:5]:
      if n in goal and abs(int(goal[n]) - self._from[n]) <= self.MAX_GOAL_LEAD_TICKS:
        self._from[n] = int(goal[n])
    self.goal = dict(self._from)

  def _hold(self, status: str) -> None:
    if status == "ok":
      # Keep the gravity-compensated setpoint. Retargeting to the measured pose
      # removes feedforward and makes the P-only servos sag after every segment.
      self.status = status
      return
    present = self.bus.sync_read("Present_Position")
    self.bus.sync_write("Goal_Position", {n: self._clamp(n, present[n]) for n in JOINT_ORDER[:5]})
    self.status = status


class AgentArm:
  def __init__(self, plant, ctrl, targets, table_z: float, speed_cm_s: float = 3.0, floor_cm: float = 0.5,
               travel_threshold_cm: float = 6.0):
    self.plant, self.bus = plant, plant.bus
    self.jmap, self.fcal = plant.jmap, plant.calib
    self.kin = ArmKinematics()
    m = self.kin.m
    self._ft = [m.geom(f"fixed_jaw_sph_tip{i}").id for i in (1, 2, 3)]
    self._mt = [m.geom(f"moving_jaw_sph_tip{i}").id for i in (1, 2, 3)]
    from so101_reach.real.robot import SimRealPlant
    fit = so101.servo_fit(so101.SIM_CONFIG_DIR / "servo_model_fit.json" if isinstance(plant, SimRealPlant) else None)
    self._kp = np.array([fit[n]["kp"] for n in JOINT_ORDER[:4]])
    self._dof = np.array([m.jnt_dofadr[m.joint(n).id] for n in JOINT_ORDER[:4]])
    self.core = MouseFollowCore(plant, ctrl, targets, source="agent")
    self.targets = targets
    self.table_z = table_z
    self.speed = speed_cm_s / 100
    self.floor = floor_cm / 100
    self.travel_threshold = travel_threshold_cm / 100
    raw = self.bus.sync_read("Present_Position")
    self.grip_goal = int(raw["gripper"])
    self.grip_torque = 250
    self.last_contact = False
    # Object in the gripper: {"name", "below_gp" (its bottom below the grip point, m), "size"}.
    self.held: dict | None = None
    self._held_floor = -1.0  # lowest allowed bottom of the held object for the current move
    self.path_guard = None  # optional (q6) -> refusal string, full-scene geometry
    self.execution_guard = None  # optional (next_q6) -> None; raises to interrupt before a command
    self.travel_controller = "rl"
    self.settle_steps = 12
    self.preferred_roll = None  # optional alternative jaw/camera arrangement for a grasp
    self.residual_tol = 0.008  # accepted final grip-point error of a move
    self.aim_guard = None  # optional (q6) -> refusal with zero clearance margin (correction aims)
    self.correction_log = None
    self.last_residual = None

  # --- Geometry ----------------------------------------------------------------------------
  def geom(self, q6: np.ndarray) -> dict:
    """Grip point, pointing, jaw closing axis, lowest jaw tip and TCP for joint angles q6."""
    tcp, a = self.kin.fk(q6)
    d = self.kin.d
    ft = np.mean([d.geom_xpos[i] for i in self._ft], axis=0)
    mt = np.mean([d.geom_xpos[i] for i in self._mt], axis=0)
    axis = mt - ft
    gp = (ft + mt) / 2 - GRIP_IN_FROM_TIPS * a
    lowest = min(d.geom_xpos[i][2] for i in self._ft + self._mt)
    # The higher of the two jaws' lowest points: both jaws must reach below an object's top to
    # capture it (the hinged moving jaw's tip rises as it opens).
    jaw_low_max = max(min(d.geom_xpos[i][2] for i in self._ft), min(d.geom_xpos[i][2] for i in self._mt))
    return {"gp": gp, "a": a, "axis": axis / max(np.linalg.norm(axis), 1e-9), "gap": float(np.linalg.norm(axis)),
            "lowest": float(lowest), "jaw_low_max": float(jaw_low_max), "tcp": tcp}

  def arm_points(self, step: float = 0.01) -> np.ndarray:
    """Points along the arm (body origins joined every `step`, plus the jaw tips), base frame,
    at the present joint angles: for removing the arm from camera detections."""
    q6, _ = self.q6()
    self.geom(q6)
    d, m = self.kin.d, self.kin.m
    chain = [m.body(n).id for n in ("shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1")]
    pts = [d.xpos[chain[0]]]
    for a, b in zip(chain[:-1], chain[1:], strict=True):
      n = max(1, int(np.linalg.norm(d.xpos[b] - d.xpos[a]) / step))
      pts += [d.xpos[a] + (d.xpos[b] - d.xpos[a]) * k / n for k in range(1, n + 1)]
    pts += [d.geom_xpos[i] for i in self._ft + self._mt]
    pts += [d.site_xpos[self.kin.site]]
    return np.array(pts)

  def _tilt_yaw(self, g: dict) -> tuple[float, float]:
    az = self.kin.azimuth(g["gp"])
    radial = np.array([np.cos(az), np.sin(az), 0.0])
    tilt = float(np.degrees(np.arctan2(g["a"] @ radial, -g["a"][2])))
    yaw = float(np.degrees(np.arctan2(g["axis"][1], g["axis"][0])))
    return tilt, (yaw + 90) % 180 - 90

  def q6(self) -> tuple[np.ndarray, dict[str, int]]:
    raw = self.bus.sync_read("Present_Position")
    return np.array([self.jmap.to_rad(n, raw[n]) for n in JOINT_ORDER]), raw

  def percent_for_gap(self, gap_m: float) -> float:
    """Smallest gripper opening (%) whose jaw-tip gap is at least `gap_m` (model geometry)."""
    q6, _ = self.q6()
    c = self.fcal["gripper"]
    for pct in range(3, 101):
      q = q6.copy()
      q[5] = self.jmap.to_rad("gripper", c.range_min + pct / 100 * (c.range_max - c.range_min))
      if self.geom(q)["gap"] >= gap_m:
        return float(pct)
    return 100.0

  def open_percent(self, ticks: float) -> float:
    c = self.fcal["gripper"]
    return float(100 * (ticks - c.range_min) / (c.range_max - c.range_min))

  def state(self) -> dict:
    q6, raw = self.q6()
    g = self.geom(q6)
    tilt, yaw = self._tilt_yaw(g)
    return {
      "grip_point_cm": [round(100 * g["gp"][0], 1), round(100 * g["gp"][1], 1), round(100 * (g["gp"][2] - self.table_z), 1)],
      "lowest_jaw_tip_h_cm": round(100 * (g["lowest"] - self.table_z), 1),
      "tilt_deg": round(tilt), "jaw_yaw_deg": round(yaw),
      "jaw_tip_gap_cm": round(100 * g["gap"], 1),
      "gripper_open_percent": round(self.open_percent(raw["gripper"])),
      "holding_signal": bool(self.last_contact and raw["gripper"] - self.grip_goal > 15),
    }

  def state_text(self) -> str:
    s = self.state()
    x, y, h = s["grip_point_cm"]
    held = ""
    if self.held and s["holding_signal"]:
      below = 100 * self.held["below_gp"]
      held = (f"\nHOLDING the {self.held['name']} ({' x '.join(f'{100 * v:.1f}' for v in self.held['size'])} cm): its bottom hangs "
              f"{below:.1f} cm below the grip point, i.e. at h={h - below:.1f} cm")
    return (f"Grip point at x={x} y={y} cm, h={h} cm above the table (lowest jaw tip h={s['lowest_jaw_tip_h_cm']} cm); "
            f"tilt {s['tilt_deg']} deg, jaw_yaw {s['jaw_yaw_deg']} deg; gripper {s['gripper_open_percent']}% open "
            f"(tips ~{s['jaw_tip_gap_cm']} cm apart)"
            + ("; HOLDING SIGNAL: the jaws are stopped short by something between them" if s["holding_signal"] else "") + held)

  # --- Planning ----------------------------------------------------------------------------
  def _check(self, q6: np.ndarray) -> str | None:
    g = self.geom(q6)
    if g["lowest"] < self.table_z + self.floor - 1e-4:
      return f"refused: a jaw tip would go to h={100 * (g['lowest'] - self.table_z):.1f} cm, below the {100 * self.floor:.1f} cm floor (table)"
    if self.held and g["gp"][2] - self.held["below_gp"] < min(self.table_z + 0.002, self._held_floor):
      return (f"refused: the held {self.held['name']} would be pushed into the table (floor): its bottom would be at "
              f"h={100 * (g['gp'][2] - self.held['below_gp'] - self.table_z):.1f} cm")
    p = g["gp"]
    lo, hi = self.targets.lo, self.targets.hi
    if np.any(p[:2] < lo[:2] - 0.005) or np.any(p[:2] > hi[:2] + 0.005):
      return (f"refused: ({100 * p[0]:.1f}, {100 * p[1]:.1f}) cm is outside the safe area x {100 * lo[0]:.0f}..{100 * hi[0]:.0f}, "
              f"y {100 * lo[1]:.0f}..{100 * hi[1]:.0f} cm")
    if self.path_guard is not None:
      return self.path_guard(q6)
    return None

  def _a_des(self, p: np.ndarray, tilt_deg: float) -> np.ndarray:
    t = np.radians(tilt_deg)
    return _rz(self.kin.azimuth(p)) @ np.array([np.sin(t), 0.0, -np.cos(t)])

  def _solve(self, p: np.ndarray, a_des: np.ndarray | None, q_init: np.ndarray) -> tuple[np.ndarray, float]:
    """Arm joints putting the grip point at p (pointing a_des as far as reachable); wrist roll
    and gripper as in q_init."""
    q6 = q_init.copy()
    weights = (0.3, 0.075, 0.02, 0.0) if a_des is not None else (0.0,)
    err = 1.0
    for w in weights:

      def res(x, w=w):
        q6[:4] = x
        g = self.geom(q6)
        r = [g["gp"] - p]
        if a_des is not None:
          r.append(w * (g["a"] - a_des))
        return np.concatenate(r)

      q6[:4] = least_squares(res, np.clip(q_init[:4], self.kin.lo, self.kin.hi), bounds=(self.kin.lo, self.kin.hi),
                             xtol=1e-8, ftol=1e-10).x
      err = float(np.linalg.norm(self.geom(q6)["gp"] - p))
      if err <= 0.002:
        break
    return q6, err

  def _roll_for_yaw(self, q6: np.ndarray, yaw_deg: float) -> float:
    lo, hi = self.kin.m.jnt_range[self.kin.m.joint("wrist_roll").id]
    best, best_cost = q6[4], 1e9
    for r in np.linspace(lo, hi, 145):
      q = q6.copy()
      q[4] = r
      _, y = self._tilt_yaw(self.geom(q))
      cost = abs((y - yaw_deg + 90) % 180 - 90) + 2.0 * abs(r - q6[4])  # prefer the smaller wrist turn
      if cost < best_cost:
        best, best_cost = r, cost
    return float(best)

  def _pose(self, p: np.ndarray, tilt: float | None, jaw_yaw: float | None, q_init: np.ndarray) -> tuple[np.ndarray, float]:
    q6 = q_init.copy()
    if jaw_yaw is not None and self.preferred_roll is not None:
      q6[4] = self.preferred_roll
    if tilt is None:  # Keep the present tilt.
      tilt, _ = self._tilt_yaw(self.geom(q_init))
    for _ in range(2 if jaw_yaw is not None else 1):
      q6, err = self._solve(p, self._a_des(p, tilt), q6)
      if jaw_yaw is not None:
        q6[4] = self._roll_for_yaw(q6, jaw_yaw)
    q6, err = self._solve(p, self._a_des(p, tilt), q6)
    return q6, err

  def _line(self, q6: np.ndarray, end: np.ndarray, tilt: float | None, jaw_yaw: float | None) -> tuple[str | None, ReplayPlan | None]:
    """Grip point along a straight line to `end`, orientation going to the requested one."""
    start = self.geom(q6)["gp"]
    # A held object may stay as low as it is now (e.g. just grasped on the table), never lower.
    self._held_floor = (start[2] - self.held["below_gp"] - 0.001) if self.held else -1.0
    dist = float(np.linalg.norm(end - start))
    q_goal, err = self._pose(end, tilt, jaw_yaw, q6)
    if err > 0.003:
      return f"refused: not reachable (IK error {1000 * err:.0f} mm)", None
    turn = np.degrees(np.abs(q_goal[:5] - q6[:5]).max())  # joints at <= 40 deg/s
    n = max(2, int(max(0.5, dist / self.speed, turn / 40) / 0.02) + 1)
    s = 0.5 - 0.5 * np.cos(np.pi * np.linspace(0, 1, n))
    t0, _ = self._tilt_yaw(self.geom(q6))
    t1, _ = self._tilt_yaw(self.geom(q_goal))  # the tilt actually reachable at the end
    qs, q = [], q6.copy()
    for k in range(n):
      q[4] = q6[4] + s[k] * (q_goal[4] - q6[4])
      p = start + s[k] * (end - start)
      q, e = self._solve(p, self._a_des(p, t0 + s[k] * (t1 - t0)), q)
      if e > 0.003:
        return f"refused: the straight path is not reachable (IK error {1000 * e:.0f} mm on the way)", None
      err = self._check(q)
      if err:
        return err + " (on the way)", None
      qs.append(q.copy())
    qs = np.array(qs)
    if np.abs(np.diff(qs[:, :4], axis=0)).max() > np.radians(4):
      # The IK switched branch on the way (orientation at the edge of reach): smooth joint move.
      jp = joint_plan(self.kin, q6, q_goal, (n - 1) * 0.02, self.grip_goal, self.grip_goal)
      if any(self._check(q) for q in jp.q):
        return "refused: no collision-free path found to there", None
      qs = jp.q
    n = len(qs)  # the joint-space fallback can have a different number of samples
    return None, ReplayPlan(np.arange(n) * 0.02, qs, np.full(n, float(self.grip_goal)), np.zeros(n), np.zeros(n),
                            np.array([self.geom(qq)["gp"] for qq in qs[:: max(1, n // 20)]]))

  def _aim_acceptable(self, q_aim: np.ndarray, q_target: np.ndarray) -> bool:
    """A sag/stiction correction commands target + measured error. The servos receive the aim
    *plus gravity feedforward*; on hardware low extended poses do not sag at all (servo loads
    low, present within 1-8 ticks of goal, gripper 13 mm above target), so the arm ends at that
    commanded pose, not at the aim's uncompensated pose. Check what the servos are sent to, with
    full margins, and the target itself; bound the overshoot below the target to 15 mm. If the
    arm really drops below its predicted path, the execution monitor stops it."""
    if self._check(q_aim) is None:
      return True
    if self._check(q_target) is not None:
      return False
    q_cmd = q_aim.copy()
    q_cmd[:4] += self._gravity_ff(q_aim)
    if self._check(q_cmd) is not None:
      return False
    overshoot = float(self.geom(q_target)["gp"][2] - self.geom(q_aim)["gp"][2])
    return overshoot <= 0.015

  def _gravity_ff(self, q: np.ndarray) -> np.ndarray:
    """Joint offsets (rad, 4 arm joints) that make the P-servos hold q against gravity."""
    d = self.kin.d
    d.qpos[self.kin.qadr] = q
    d.qvel[:] = 0
    mujoco.mj_forward(self.kin.m, d)
    return d.qfrc_bias[self._dof] / self._kp

  def _with_ff(self, plan: ReplayPlan) -> tuple[ReplayPlan, float]:
    ff = np.array([self._gravity_ff(q) for q in plan.q])
    q = plan.q.copy()
    q[:, :4] += ff
    return ReplayPlan(plan.t, q, plan.gripper_goal, plan.pos_err, plan.point_err_deg, plan.targets), float(np.abs(ff).max())

  def track_points(self, q6: np.ndarray) -> np.ndarray:
    """Collision-relevant points (distal link origins, camera mount, jaw tips) for path tracking."""
    self.kin.fk(q6)
    d, m = self.kin.d, self.kin.m
    if not hasattr(self, "_track_bodies"):
      self._track_bodies = [m.body(n).id for n in ("lower_arm", "wrist", "gripper", "moving_jaw_so101_v1", "camera_mount")]
    return np.concatenate([d.xpos[self._track_bodies], d.geom_xpos[self._ft + self._mt]])

  def _run(self, plan: ReplayPlan, approach_s: float = 0.2, predicted: np.ndarray | None = None) -> str:
    """predicted: expected physical joint path, same length as plan.q (default: plan.q)."""
    plan.gripper_goal[:] = self.grip_goal
    # The servos receive nominal + gravity feedforward, and P-servo sag returns the arm to the
    # nominal pose. Collision checks use the predicted physical path; the monitor measures the
    # actual deviation from that path every tick and adds it to the required clearance.
    nominal = plan.q.copy() if predicted is None else np.asarray(predicted)[
      np.round(np.linspace(0, len(predicted) - 1, len(plan.q))).astype(int)]
    path_pts = np.array([self.track_points(q) for q in nominal]) if self.execution_guard is not None else None
    plan, ffmax = self._with_ff(plan)
    player = _ArmPlayer(self.bus, plan, self.jmap, self.fcal, approach_s=approach_s, gripper_torque=self.grip_torque,
                        stuck_rad=0.15 + ffmax, stuck_s=0.3)
    while not player.done:
      if self.execution_guard is not None:
        t = max(0, player._k + 2 - player._n_approach) * player.dt
        i = min(len(plan.q) - 1, max(0, int(np.searchsorted(plan.t, t))))
        self.execution_guard(nominal[i], predicted_path=path_pts[max(0, i - 50):i + 1])
      player.step()
      self.plant.wait()
    return player.status

  def _play(self, plan: ReplayPlan, target: np.ndarray | None = None) -> str:
    """Play a joint plan; then, if `target` (grip point) is given, correct the remaining error
    (servo sag under gravity: the STS3215 position loop has no integral term) by re-aiming at
    target + error, up to 3 times. No correction after a blocked move (contact)."""
    if self._run(plan) != "ok":
      return "stopped: an arm joint could not follow (blocked by contact?) - the arm holds where it is"
    if target is None:
      return "ok"
    for _ in range(self.settle_steps):
      self.plant.wait()
    aim, last = target.copy(), None
    q_target = plan.q[-1].copy()  # nominal pose for `target`: where a correction should physically end
    self.correction_log = []
    for _ in range(3):
      q6, _ = self.q6()
      e = target - self.geom(q6)["gp"]
      self.correction_log.append({"error_mm": [round(1000 * float(x), 1) for x in e]})
      if np.linalg.norm(e) < 0.004 or np.linalg.norm(e) > 0.06 or (last is not None and np.linalg.norm(e) > 0.8 * last):
        self.correction_log[-1]["stop"] = "small" if np.linalg.norm(e) < 0.004 else "too large" if np.linalg.norm(e) > 0.06 else "not improving"
        break
      last = float(np.linalg.norm(e))
      aim = aim + e
      q_aim, err = self._solve(aim, None, plan.q[-1].copy())
      if err > 0.003 or not self._aim_acceptable(q_aim, q_target):
        self.correction_log[-1]["stop"] = f"aim refused: ik {1000 * err:.1f} mm / {self._check(q_aim)}"
        break
      pl = joint_plan(self.kin, plan.q[-1], q_aim, 0.5, self.grip_goal, self.grip_goal)
      pl.q[:, 4:] = q_aim[4:]
      # The command aims past the target to cancel residual sag; physically the arm moves from
      # its measured pose to the target pose. That is the path the monitor must check.
      expect = joint_plan(self.kin, q6, q_target, 0.5, self.grip_goal, self.grip_goal).q
      expect[:, 4:] = q_aim[4:]
      plan = pl
      if self._run(pl, approach_s=0.1, predicted=expect) != "ok":
        return "stopped: an arm joint could not follow (blocked by contact?) - the arm holds where it is"
      for _ in range(20):
        self.plant.wait()
    residual = float(np.linalg.norm(target - self.geom(self.q6()[0])["gp"]))
    for _ in range(25):  # measured on hardware: a 9.6 mm residual settled to ~2 mm within ~0.5 s
      if residual <= self.residual_tol:
        break
      self.plant.wait()
      residual = float(np.linalg.norm(target - self.geom(self.q6()[0])["gp"]))
    self.last_residual = residual
    return "ok" if residual <= self.residual_tol else f"stopped: final grip-point error {1000 * residual:.1f} mm"

  def _travel(self, tcp_target: np.ndarray) -> str:
    """RL reach policy: moves the TCP (closed-jaw tip) to tcp_target (at least the policy's
    minimum travel height)."""
    tcp_target = np.array([tcp_target[0], tcp_target[1], max(tcp_target[2], self.targets.lo[2])])
    mv = PolicyMove(self.core, tcp_target, tol=0.01)
    if self.execution_guard is not None:
      def guard(jt):
        q6, _ = self.q6()
        q6[:4] = jt
        self.execution_guard(q6)
      mv.command_guard = guard
    while not mv.done:
      mv.step()
      self.plant.wait()
    if mv.status != "ok":
      return f"travel {mv.status}"
    # PolicyMove ends by holding goal = present, which drops the gravity compensation implicit in
    # the policy's commands: re-hold with feedforward and settle before the next IK segment plans.
    self._compensated_hold()
    return "ok"

  def _compensated_hold(self, settle_ticks: int = 10) -> None:
    q6, raw = self.q6()
    ff = self._gravity_ff(q6)
    goal = {n: self.plant._clamp(n, self.jmap.to_ticks(n, q6[j] + ff[j])) for j, n in enumerate(JOINT_ORDER[:4])}
    goal["wrist_roll"] = self.plant._clamp("wrist_roll", raw["wrist_roll"])
    self.bus.sync_write("Goal_Position", goal)
    for _ in range(settle_ticks):
      self.plant.wait()

  def lowest_h(self, x_m: float, y_m: float, tilt: float | None, jaw_yaw: float | None, q6: np.ndarray) -> float | None:
    """Lowest grip-point height (cm above the table) at (x, y) that keeps the jaw tips above
    the floor, with the given orientation and the present opening."""
    for h in np.arange(0.0, 15.0, 0.2):
      p = np.array([x_m, y_m, self.table_z + h / 100])
      q, err = self._pose(p, tilt, jaw_yaw, q6)
      if err <= 0.003 and self._check(q) is None:
        return float(h)
    return None

  def _rise_for_travel(self, q6: np.ndarray) -> str:
    here = self.geom(q6)
    travel_z = self.targets.lo[2] + (here["gp"][2] - here["tcp"][2])
    if here["gp"][2] >= travel_z - 0.01:
      return "ok"
    err, plan = self._line(q6, np.array([*here["gp"][:2], travel_z]), None, None)
    if err:
      return err + " (while rising to travel height)"
    st = self._play(plan)
    return st if st == "ok" else st + " (while rising to travel height)"

  # --- Tools -------------------------------------------------------------------------------
  def move_to(self, x: float, y: float, h: float, tilt: float | None = None, jaw_yaw: float | None = None) -> str:
    p = np.array([x / 100, y / 100, self.table_z + h / 100])
    q6, _ = self.q6()
    q_goal, err = self._pose(p, tilt, jaw_yaw, q6)
    if err > 0.003:
      return f"refused: position not reachable (IK error {1000 * err:.0f} mm)"
    bad = self._check(q_goal)
    if bad:
      if "floor" in bad:
        h_min = self.lowest_h(p[0], p[1], tilt, jaw_yaw, q6)
        if h_min is not None:
          bad += f"; the lowest allowed h here with this orientation and opening is {h_min:.1f} cm"
      return bad
    notes = []
    if tilt is not None:
      got, _ = self._tilt_yaw(self.geom(q_goal))
      if abs(got - tilt) > 8:
        notes.append(f"tilt {tilt:.0f} deg is not reachable there, using {got:.0f}")
    if self.travel_controller == "rl" and np.linalg.norm(p - self.geom(q6)["gp"]) > self.travel_threshold:
      # Long move: rise straight up to the travel height, RL travel, then straight in.
      st = self._rise_for_travel(q6)
      if st != "ok":
        return st
      g = self.geom(q_goal)
      st = self._travel(p + (g["tcp"] - g["gp"]))
      if st != "ok":
        return st + " - the arm holds where it is"
      q6, _ = self.q6()
    err, plan = self._line(q6, p, tilt, jaw_yaw)
    if err:
      return err
    return self._play(plan, p) + (f" ({'; '.join(notes)})" if notes else "")

  def move_by(self, dx: float, dy: float, dz: float) -> str:
    q6, _ = self.q6()
    start = self.geom(q6)["gp"]
    end = start + np.array([dx, dy, dz]) / 100
    if np.linalg.norm(end - start) > 0.10:
      return "refused: move_by is for fine moves (<= 10 cm); use move_to"
    err, plan = self._line(q6, end, None, None)
    if err:
      return err
    return self._play(plan, end)

  def set_orientation(self, tilt: float | None = None, jaw_yaw: float | None = None) -> str:
    q6, _ = self.q6()
    here = self.geom(q6)["gp"]
    err, plan = self._line(q6, here, tilt, jaw_yaw)
    if err:
      return err
    st = self._play(plan, here)
    t, y = self._tilt_yaw(self.geom(self.q6()[0]))
    return st + f" (now tilt {t:.0f} deg, jaw_yaw {y:.0f} deg)"

  def gripper(self, open_percent: float, force: str = "medium") -> str:
    c = self.fcal["gripper"]
    open_percent = float(np.clip(open_percent, 0, 100))
    target = int(c.range_min + open_percent / 100 * (c.range_max - c.range_min))
    torque, squeeze = FORCES.get(force, FORCES["medium"])
    self.bus.write("gripper", "Torque_Limit", torque)
    start = self.bus.sync_read("Present_Position")["gripper"]
    closing = target < start
    n = max(15, int(abs(target - start) / (30 if closing else 12)))  # open gently: no flick on release
    hist, contact, goal, pres = [], False, start, start
    for k in range(1, n + 150):  # ramp, then up to 3 s for the jaws to arrive or stop
      goal = int(start + (target - start) * min(k / n, 1.0))
      if self.execution_guard is not None:
        q6, _ = self.q6()
        q6[5] = self.jmap.to_rad("gripper", goal)
        self.execution_guard(q6)
      self.bus.sync_write("Goal_Position", {"gripper": goal})
      self.plant.wait()
      pres = self.bus.sync_read("Present_Position")["gripper"]
      hist.append(pres)
      still = len(hist) > 10 and abs(hist[-11] - pres) < 4
      # Contact: closing, the jaws stopped short of the goal, not at the closed end.
      if closing and k >= n and still and pres - goal > 40 and pres > c.range_min + 70:
        contact = True
        break
      if k >= n and (abs(pres - goal) < 15 or still):
        break
    if contact:
      goal = int(max(c.range_min, pres - squeeze))
      self.bus.sync_write("Goal_Position", {"gripper": goal})
      for _ in range(10):
        self.plant.wait()
    self.grip_goal, self.grip_torque, self.last_contact = goal, torque, contact
    if not closing:
      self.held = None
    pres = self.bus.sync_read("Present_Position")["gripper"]
    if contact:
      return f"closed on something: the jaws stopped at {self.open_percent(pres):.0f}% open ({force} grip)"
    if closing and open_percent < 5:
      return f"closed fully ({self.open_percent(pres):.0f}%) - NOTHING was caught between the jaws"
    return f"ok: gripper at {self.open_percent(pres):.0f}% open"

  def home(self) -> str:
    home = np.array([so101.HOME_JOINT_POS[n] for n in so101.ARM_JOINTS] + [self.jmap.to_rad("wrist_roll", 2048), 0.0])
    q6, _ = self.q6()
    home[5] = q6[5]
    g_home = self.geom(home)
    park = np.radians([CLEAR_VIEW_DEG[n] for n in JOINT_ORDER[:4]])
    if np.abs(q6[:4] - park).max() < np.radians(15):  # leaving the parking pose: checked joint path
      jp = joint_plan(self.kin, q6, home, max(2.0, np.degrees(np.abs(home[:5] - q6[:5]).max()) / 30), self.grip_goal, self.grip_goal)
      if all(self._links_clear(q) for q in jp.q):
        return "ok" if self._run(jp) == "ok" else "stopped: an arm joint could not follow (blocked by contact?) - the arm holds where it is"
    if self.travel_controller == "rl" and np.linalg.norm(self.geom(q6)["tcp"] - g_home["tcp"]) > self.travel_threshold:
      st = self._rise_for_travel(q6)
      if st != "ok":
        return st
      st = self._travel(g_home["tcp"])
      if st != "ok":
        return st
      q6, _ = self.q6()
    return "ok" if self._run(joint_plan(self.kin, q6, home, 2.0, self.grip_goal, self.grip_goal)) == "ok" else \
      "stopped: an arm joint could not follow (blocked by contact?) - the arm holds where it is"

  def _links_clear(self, q6: np.ndarray, min_h: float = 0.04) -> bool:
    """All links beyond the shoulder at least `min_h` above the table and left of the wall limit."""
    self.geom(q6)
    d, m = self.kin.d, self.kin.m
    if not hasattr(self, "_link_geoms"):
      moving = {m.body(n).id for n in ("upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1", "camera_mount")}
      self._link_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in moving and (m.geom_contype[g] or m.geom_conaffinity[g])]
    p = d.geom_xpos[self._link_geoms]
    return bool(p[:, 2].min() >= self.table_z + min_h and p[:, 1].max() <= self.targets.hi[1] + 0.03)

  def observe_pose(self, table_z: float) -> str:
    """Rise straight up if low, then a monitored joint move to OBSERVE_DEG (gripper kept)."""
    q6, _ = self.q6()
    gp = self.geom(q6)["gp"]
    if gp[2] - table_z < 0.09:
      err, plan = self._line(q6, np.array([gp[0], gp[1], table_z + 0.11]), None, None)
      if err:
        return err + " (while rising to observe)"
      st = self._play(plan)
      if st != "ok":
        return st
      q6, _ = self.q6()
    target = q6.copy()
    target[:4] = np.radians([OBSERVE_DEG[n] for n in JOINT_ORDER[:4]])
    target[4] = self.jmap.to_rad("wrist_roll", 2048)
    dur = max(3.0, float(np.degrees(np.abs(target[:5] - q6[:5]).max())) / 20)
    st = self._run(joint_plan(self.kin, q6, target, dur, self.grip_goal, self.grip_goal))
    return "ok" if st == "ok" else "stopped: " + st

  def clear_view(self) -> str:
    """Park the arm out of the head camera's view of the table (keeps the gripper as it is)."""
    q6, _ = self.q6()
    st = self._rise_for_travel(q6)
    if st != "ok":
      return st
    q6, _ = self.q6()
    park = q6.copy()
    park[:4] = np.radians([CLEAR_VIEW_DEG[n] for n in JOINT_ORDER[:4]])
    park[4] = self.jmap.to_rad("wrist_roll", 2048)
    plan = joint_plan(self.kin, q6, park, max(2.0, np.degrees(np.abs(park[:5] - q6[:5]).max()) / 30), self.grip_goal, self.grip_goal)
    for q in plan.q:
      if not self._links_clear(q):
        return "refused: no safe path to the parking pose from here - move up / away first"
    st = self._run(plan)
    return "ok: the arm is parked out of the head camera's view" if st == "ok" else \
      "stopped: an arm joint could not follow (blocked by contact?) - the arm holds where it is"

  def wait(self, seconds: float) -> str:
    for _ in range(int(min(seconds, 5.0) / 0.02)):
      self.plant.wait()
    return "ok"
