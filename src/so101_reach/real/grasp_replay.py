"""Replay a taught grasp (data/grasp_demos/*.npz) at a new orange position.

Retargeting: every demo sample i gives the TCP position and the gripper pointing direction.
For a new orange at azimuth psi' (demo: psi) about the pan axis:

  target_i    = orange' + Rz(psi' - psi) (tcp_i - orange_demo)
  pointing_i' = Rz(psi' - psi) pointing_i

and the four arm joints are solved by bounded least squares (position first, pointing with
a small weight), warm-started along the path; wrist roll and gripper are replayed as taught.
Execution streams the joint goals at the demo timing (optionally faster), rate-limited, with
a blocked-joint stop, and ends with a gentle gripper hold (no stall).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from so101_reach.real.feetech_bus import JOINT_ORDER, MotorCalib
from so101_reach.real.joint_map import JointMap
from so101_reach.sim import so101

DEMO_DIR: Path = so101.PROJECT_ROOT / "data" / "grasp_demos"


def demo_method(path: Path) -> str:
  """Grasp method a demo was taught as (demos from before methods existed are "orange")."""
  meta = path.with_suffix(".json")
  return json.loads(meta.read_text()).get("method", "orange") if meta.exists() else "orange"


def list_methods() -> dict[str, list[Path]]:
  out: dict[str, list[Path]] = {}
  for d in sorted(DEMO_DIR.glob("grasp_*.npz")):
    out.setdefault(demo_method(d), []).append(d)
  return out


def latest_demo(method: str | None = None) -> Path:
  """Newest demo, or the newest one taught as `method`."""
  demos = sorted(DEMO_DIR.glob("grasp_*.npz"))
  if method is not None:
    demos = [d for d in demos if demo_method(d) == method]
  if not demos:
    known = ", ".join(list_methods()) or "none"
    raise FileNotFoundError(f"no grasp demo for method {method!r} in {DEMO_DIR} (taught methods: {known}) - "
                            f"run scripts/teach_grasp.py --method {method or 'NAME'}")
  return demos[-1]


@dataclass
class GraspDemo:
  t: np.ndarray  # (N,) s
  q: np.ndarray  # (N, 6) rad, joint order JOINT_ORDER
  tcp: np.ndarray  # (N, 3) base frame
  gripper_goal: np.ndarray  # (N,) follower raw ticks (as commanded)
  orange: np.ndarray  # (3,)
  meta: dict

  @staticmethod
  def load(path: Path) -> GraspDemo:
    d = np.load(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    return GraspDemo(d["t"], d["q"], d["tcp"], d["follower_goal"][:, 5].astype(float), np.array(meta["orange_center_base"]), meta)


class ArmKinematics:
  def __init__(self):
    self.m = so101.make_cpu_model()
    self.d = mujoco.MjData(self.m)
    self.qadr = np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in JOINT_ORDER])
    self.site = self.m.site(so101.TCP_SITE).id
    self.body = self.m.body("gripper").id
    jid = self.m.joint("shoulder_pan").id
    mujoco.mj_kinematics(self.m, self.d)
    self.pan_anchor = self.d.xanchor[jid][:2].copy()
    lim = np.array([self.m.jnt_range[self.m.joint(n).id] for n in JOINT_ORDER[:4]])
    self.lo, self.hi = lim[:, 0], lim[:, 1]

  def fk(self, q6: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """TCP position and gripper pointing direction (gripper body -> TCP, unit)."""
    self.d.qpos[self.qadr] = q6
    mujoco.mj_kinematics(self.m, self.d)
    tcp = self.d.site_xpos[self.site].copy()
    a = tcp - self.d.xpos[self.body]
    return tcp, a / np.linalg.norm(a)

  def azimuth(self, p: np.ndarray) -> float:
    v = p[:2] - self.pan_anchor
    return float(np.arctan2(v[1], v[0]))


def _rz(a: float) -> np.ndarray:
  c, s = np.cos(a), np.sin(a)
  return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


@dataclass
class ReplayPlan:
  t: np.ndarray
  q: np.ndarray  # (N, 6) rad
  gripper_goal: np.ndarray  # (N,) raw ticks
  pos_err: np.ndarray  # (N,) m, IK position residual
  point_err_deg: np.ndarray  # (N,) pointing deviation
  targets: np.ndarray  # (N, 3)


def retarget(demo: GraspDemo, orange_new: np.ndarray, q_start: np.ndarray | None = None,
             kin: ArmKinematics | None = None, w_point: float = 0.05, stride: int = 1) -> ReplayPlan:
  kin = kin or ArmKinematics()
  idx = np.arange(0, len(demo.t), stride)
  rot = _rz(kin.azimuth(orange_new) - kin.azimuth(demo.orange))
  q_prev = (q_start if q_start is not None else demo.q[0])[:4].copy()
  qs, perr, aerr, tg = [], [], [], []
  for i in idx:
    q6 = demo.q[i].copy()
    _, a_demo = kin.fk(q6)
    tcp_i, _ = kin.fk(q6)  # from the recorded joints (current joint map), not the stored TCP
    target = orange_new + rot @ (tcp_i - demo.orange)
    a_des = rot @ a_demo

    def res(x, q6=q6, target=target, a_des=a_des):
      q6[:4] = x
      p, a = kin.fk(q6)
      return np.concatenate([p - target, w_point * (a - a_des)])

    sol = least_squares(res, np.clip(q_prev, kin.lo, kin.hi), bounds=(kin.lo, kin.hi), xtol=1e-8, ftol=1e-10)
    q6[:4] = sol.x
    p, a = kin.fk(q6)
    qs.append(q6.copy())
    perr.append(np.linalg.norm(p - target))
    aerr.append(np.degrees(np.arccos(np.clip(a @ a_des, -1, 1))))
    tg.append(target)
    q_prev = sol.x
  return ReplayPlan(demo.t[idx] - demo.t[idx[0]], np.array(qs), demo.gripper_goal[idx], np.array(perr), np.array(aerr), np.array(tg))


def _plan_from_q(kin: ArmKinematics, t: np.ndarray, q: np.ndarray, grip: np.ndarray, targets: np.ndarray | None = None) -> ReplayPlan:
  tcp = np.array([kin.fk(qi)[0] for qi in q])
  return ReplayPlan(t, q, grip, np.zeros(len(t)), np.zeros(len(t)), tcp if targets is None else targets)


def line_plan(kin: ArmKinematics, q6: np.ndarray, end: np.ndarray, duration: float, gripper_goal: float,
              dt: float = 0.02, w_point: float = 0.2, pos_tol: float = 0.001) -> ReplayPlan:
  """Straight TCP line from FK(q6) to `end` (smooth start/stop). The gripper pointing is kept
  as at the start, turned with the base (pan) - as far as the position allows: where that
  pointing is out of reach (far targets) the pointing weight is reduced until the position
  is met within `pos_tol`."""
  start, a0 = kin.fk(q6)
  az0 = kin.azimuth(start)
  n = max(2, int(duration / dt) + 1)
  s = 0.5 - 0.5 * np.cos(np.pi * np.linspace(0, 1, n))
  q = np.repeat(q6[None].astype(float), n, axis=0)
  x, perr, tg = q6[:4].copy(), [], []
  for k in range(n):
    target = start + s[k] * (end - start)
    a_des = _rz(kin.azimuth(target) - az0) @ a0
    for w in (w_point, w_point / 4, w_point / 16, 0.0):

      def res(xx, k=k, target=target, a_des=a_des, w=w):
        q[k, :4] = xx
        p, a = kin.fk(q[k])
        return np.concatenate([p - target, w * (a - a_des)])

      sol = least_squares(res, np.clip(x, kin.lo, kin.hi), bounds=(kin.lo, kin.hi), xtol=1e-8, ftol=1e-10).x
      q[k, :4] = sol
      err = np.linalg.norm(kin.fk(q[k])[0] - target)
      if err <= pos_tol:
        break
    x = sol
    perr.append(err)
    tg.append(target)
  plan = _plan_from_q(kin, np.arange(n) * dt, q, np.full(n, float(gripper_goal)), np.array(tg))
  plan.pos_err = np.array(perr)
  return plan


def joint_plan(kin: ArmKinematics, q6_from: np.ndarray, q6_to: np.ndarray, duration: float, grip_from: float, grip_to: float,
               dt: float = 0.02) -> ReplayPlan:
  """Smooth joint-space interpolation (cosine profile) including the gripper goal."""
  n = max(2, int(duration / dt) + 1)
  s = (0.5 - 0.5 * np.cos(np.pi * np.linspace(0, 1, n)))[:, None]
  q = q6_from[None] + s * (q6_to - q6_from)[None]
  return _plan_from_q(kin, np.arange(n) * dt, q, grip_from + s[:, 0] * (grip_to - grip_from))


class PlanPlayer:
  """Streams a ReplayPlan to the follower, one `step()` per 20 ms control tick.

  Phases: approach (smooth joint move from the present pose to the plan start) -> play (demo
  timing / speed, goals rate-limited) -> hold. The play phase stops early if an arm joint lags
  its goal by more than `stuck_rad` for `stuck_s`. Hold: arm at its present pose; if the
  gripper was stopped short of its closing goal (it holds an object) its goal goes just inside
  the contact position at reduced torque, so the STS3215 never sits in a stall.
  """

  def __init__(self, bus, plan: ReplayPlan, jmap: JointMap, fcal: dict[str, MotorCalib], speed: float = 1.0,
               approach_s: float = 1.5, max_step: int = 40, gripper_torque: int = 350, hold_torque: int = 250,
               hold_squeeze: int = 30, stuck_rad: float = 0.35, stuck_s: float = 0.5, dt: float = 0.02):
    self.bus, self.plan, self.jmap, self.fcal = bus, plan, jmap, fcal
    self.speed, self.max_step, self.dt = speed, max_step, dt
    self.hold_torque, self.hold_squeeze = hold_torque, hold_squeeze
    self.stuck_rad, self.stuck_s = stuck_rad, stuck_s
    self.status = "running"
    self.gripper_gap = 0
    self.log: list = []
    bus.write("gripper", "Torque_Limit", gripper_torque)
    present = bus.sync_read("Present_Position")
    self._from = {n: int(present[n]) for n in JOINT_ORDER}
    self._first = self._goal_at(0)
    self._n_approach = max(1, int(approach_s / dt))
    self._k = 0
    self._stuck = 0
    self.goal = dict(self._from)

  def _clamp(self, n: str, v: float) -> int:
    return int(np.clip(round(v), self.fcal[n].range_min, self.fcal[n].range_max))

  def _goal_at(self, k: int) -> dict[str, int]:
    g = {n: self._clamp(n, self.jmap.to_ticks(n, self.plan.q[k, j])) for j, n in enumerate(JOINT_ORDER[:5])}
    g["gripper"] = self._clamp("gripper", self.plan.gripper_goal[k])
    return g

  @property
  def done(self) -> bool:
    return self.status != "running"

  @property
  def progress(self) -> float:
    total = self._n_approach + self.plan.t[-1] / self.speed / self.dt
    return min(1.0, self._k / max(total, 1))

  def step(self) -> None:
    if self.done:
      return
    self._k += 1
    if self._k <= self._n_approach:
      a = 0.5 - 0.5 * np.cos(np.pi * self._k / self._n_approach)
      self.goal = {n: int(round(self._from[n] + a * (self._first[n] - self._from[n]))) for n in JOINT_ORDER}
      self.bus.sync_write("Goal_Position", self.goal)
      return
    t = (self._k - self._n_approach) * self.dt
    if t > self.plan.t[-1] / self.speed:
      self._hold("ok")
      return
    k = int(np.searchsorted(self.plan.t / self.speed, t, side="right") - 1)
    target = self._goal_at(max(k, 0))
    self.goal = {n: self.goal[n] + int(np.clip(target[n] - self.goal[n], -self.max_step, self.max_step)) for n in JOINT_ORDER}
    self.bus.sync_write("Goal_Position", self.goal)
    present = self.bus.sync_read("Present_Position")
    lag = max(abs(self.jmap.to_rad(n, self.goal[n]) - self.jmap.to_rad(n, present[n])) for n in JOINT_ORDER[:4])
    self._stuck = self._stuck + 1 if lag > self.stuck_rad else 0
    self.log.append((t, [present[n] for n in JOINT_ORDER], [self.goal[n] for n in JOINT_ORDER]))
    if self._stuck * self.dt >= self.stuck_s:
      self._hold(f"stopped: arm not following the plan (lag {np.degrees(lag):.0f} deg) at t={t:.2f} s")

  def _hold(self, status: str) -> None:
    present = self.bus.sync_read("Present_Position")
    hold = {n: self._clamp(n, present[n]) for n in JOINT_ORDER}
    self.gripper_gap = int(present["gripper"] - self.goal["gripper"])  # > 0: stopped short of its goal
    if self.goal["gripper"] < present["gripper"] - 20:  # Closing but blocked: holding an object.
      hold["gripper"] = self._clamp("gripper", present["gripper"] - self.hold_squeeze)
    else:
      hold["gripper"] = self.goal["gripper"]
    self.bus.write("gripper", "Torque_Limit", self.hold_torque)
    self.bus.sync_write("Goal_Position", hold)
    self.status = status


def execute(bus, plan: ReplayPlan, jmap: JointMap, fcal: dict[str, MotorCalib], **kw) -> str:
  """Blocking PlanPlayer run at 50 Hz. Returns 'ok' or the stop reason; the arm holds at the end."""
  player = PlanPlayer(bus, plan, jmap, fcal, **kw)
  deadline = time.perf_counter()
  while not player.done:
    player.step()
    deadline += player.dt
    time.sleep(max(0.0, deadline - time.perf_counter()))
  return player.status


def home_q6() -> np.ndarray:
  pose = {**so101.HOME_JOINT_POS, **so101.FIXED_JOINT_TARGETS}
  return np.array([pose[n] for n in JOINT_ORDER])


def place_sequence(kin: ArmKinematics, jmap: JointMap, demo: GraspDemo, q6: np.ndarray, grip_hold: float, orange_z: float,
                   lower_speed: float = 0.02, rise: float = 0.06) -> list[tuple[str, ReplayPlan]]:
  """Put a held orange down where it is, open, rise, go home. The place height is the demo's
  grasp height (TCP above the orange centre) + 5 mm."""
  tcp, _ = kin.fk(q6)
  z_place = orange_z + (demo.tcp[:, 2].min() - demo.orange[2]) + 0.005
  down = np.array([tcp[0], tcp[1], z_place])
  lower = line_plan(kin, q6, down, max(1.0, (tcp[2] - z_place) / lower_speed), grip_hold)
  q = lower.q[-1]
  grip_open = float(demo.gripper_goal.max())
  opening = joint_plan(kin, q, q, 1.0, grip_hold, grip_open)
  up = line_plan(kin, q, down + np.array([0, 0, rise]), rise / lower_speed, grip_open)
  q = up.q[-1]
  home = home_q6()
  homing = joint_plan(kin, q, home, 3.0, grip_open, jmap.to_ticks("gripper", home[5]))
  return [("lower", lower), ("open", opening), ("rise", up), ("home", homing)]


def tilt_deg(kin: ArmKinematics, q6: np.ndarray) -> float:
  """Angle between the gripper pointing and straight down."""
  return float(np.degrees(np.arccos(np.clip(-kin.fk(q6)[1][2], -1, 1))))


def drop_geometry(kin: ArmKinematics, cont, tcp: np.ndarray, obj_radius: float, site_lo: np.ndarray, site_hi: np.ndarray,
                  carry_clear: float = 0.06, drop_clear: float = 0.03, wall_margin: float = 0.015) -> tuple[np.ndarray, float, float]:
  """Drop point (xy), carry height and drop height (TCP z) for a held object of `obj_radius`."""
  hang = max(obj_radius - 0.008, 0.0)  # Object bottom below the TCP (grasped ~8 mm below its centre).
  drop = cont.drop_point(kin.pan_anchor, obj_radius + wall_margin)
  drop = np.clip(drop, site_lo[:2] + 0.005, site_hi[:2] - 0.005)
  if not cont.contains(drop, grow=-0.01):
    raise ValueError(f"no drop point inside the {cont.name} within the site limits (configs/real_workspace.json)")
  return drop, max(tcp[2], cont.rim_z + hang + carry_clear), cont.rim_z + hang + drop_clear


@dataclass
class PolicyTarget:
  """Queue item: move the TCP to `target` with the RL reach policy."""

  target: np.ndarray


@dataclass
class Deferred:
  """Queue item: a plan built when its phase starts, from the arm's pose then (q6, raw ticks)."""

  build: object  # callable(q6, raw) -> ReplayPlan; raises ValueError if not possible


class PolicyMove:
  """Moves the TCP to a target with the RL reach policy (the same controller that follows the
  mouse / the object), with PlanPlayer's interface. The wrist roll and gripper keep their
  present goals (a held object stays gripped). Ends `ok` once within `tol` for `settle_s`;
  stops (arm holds) if a joint is blocked for `stuck_s` or after `timeout_s`."""

  def __init__(self, core, target: np.ndarray, tol: float = 0.008, settle_s: float = 0.3, timeout_s: float = 10.0,
               stuck_s: float = 0.5):
    self.core = core
    self.target, self.note = core.targets(np.asarray(target, float))  # Site limits, min height, workspace.
    self.tol, self.settle_s, self.timeout_s, self.stuck_s = tol, settle_s, timeout_s, stuck_s
    self.status, self.gripper_gap, self.dt = "running", 0, core.plant.dt
    q = core.plant.read_q()
    core.ctrl.reset(q)
    self.err0 = self.err = float(np.linalg.norm(core.ctrl.kin.tcp(q) - self.target))
    self._k = self._settled = self._stuck = 0
    self._q_hist: list[np.ndarray] = []
    self.command_guard = None

  @property
  def done(self) -> bool:
    return self.status != "running"

  @property
  def progress(self) -> float:
    return float(np.clip(1.0 - self.err / max(self.err0, 1e-6), 0.0, 1.0))

  def _hold_arm(self) -> None:
    plant = self.core.plant
    raw = plant.last_raw or plant.bus.sync_read("Present_Position")
    plant.bus.sync_write("Goal_Position", {n: plant._clamp(n, raw[n]) for n in so101.ARM_JOINTS})

  def step(self) -> None:
    if self.done:
      return
    plant, ctrl = self.core.plant, self.core.ctrl
    q = plant.read_q()
    if plant.fixed:
      ctrl.kin.set_fixed(plant.fixed)
    jt, tcp = ctrl.step(q, self.target, plant.dt)
    self.err = float(np.linalg.norm(tcp - self.target))
    self._k += 1
    lag = np.abs(jt - q) >= ctrl.cfg.max_lead - 1e-3
    # Blocked = a joint at the lead limit that has also stopped moving (< 0.6 deg in 0.25 s); at
    # high speed the command legitimately runs ahead of a moving servo.
    self._q_hist = (self._q_hist + [q])[-13:]
    moving = np.abs(q - self._q_hist[0]) > np.radians(0.6) if len(self._q_hist) == 13 else np.ones_like(lag)
    lag = lag & ~moving
    self._stuck = self._stuck + 1 if lag.any() else 0
    if self._stuck * self.dt >= self.stuck_s:
      joints = [n for n, b in zip(so101.ARM_JOINTS, lag, strict=True) if b]
      self._hold_arm()
      self.status = f"stopped: {', '.join(joints)} not following (blocked?)"
      return
    if self.command_guard is not None:
      self.command_guard(jt)
    plant.command(jt)
    self.core.q, self.core.tcp, self.core.target = q, tcp, self.target
    self._settled = self._settled + 1 if self.err < self.tol else 0
    if self._settled * self.dt >= self.settle_s:
      self._hold_arm()
      self.status = "ok"
    elif self._k * self.dt >= self.timeout_s:
      self._hold_arm()
      self.status = f"stopped: target not reached in {self.timeout_s:.0f} s (error {self.err * 1e3:.0f} mm)"


def container_steps(kin: ArmKinematics, jmap: JointMap, cont, q6: np.ndarray, grip_hold: float, grip_open: float,
                    obj_radius: float, site_lo: np.ndarray, site_hi: np.ndarray, v_vert: float = 0.03, max_tilt: float = 70.0,
                    max_ik_err: float = 0.003) -> list[tuple[str, object]]:
  """container_sequence with the long move done by the RL reach policy:
  lift (IK, straight up) -> carry (RL policy to above the drop point) -> lower (IK, straight down,
  planned from where the policy stopped) -> open -> rise (the lowering path backwards) -> home.
  The full IK sequence is planned first as a feasibility check (ValueError: nothing moves)."""
  check = container_sequence(kin, jmap, cont, q6, grip_hold, grip_open, obj_radius, site_lo, site_hi,
                             v_vert=v_vert, max_tilt=max_tilt, max_ik_err=max_ik_err)
  tcp, _ = kin.fk(q6)
  drop, carry_z, drop_z = drop_geometry(kin, cont, tcp, obj_radius, site_lo, site_hi)
  memo: dict = {}

  def lower(q, raw):
    start, _ = kin.fk(q)
    end = np.array([drop[0], drop[1], drop_z])
    plan = line_plan(kin, q, end, max(0.6, np.linalg.norm(end - start) / v_vert), grip_hold)
    if plan.pos_err.max() > max_ik_err:
      raise ValueError(f"lower: out of reach from the carry pose (IK error {plan.pos_err.max() * 1e3:.0f} mm)")
    if not cont.contains(start[:2], grow=-0.005):
      raise ValueError(f"the policy stopped outside the {cont.name} ({start[0]:.3f}, {start[1]:+.3f})")
    tilt = tilt_deg(kin, plan.q[-1])
    if tilt > max_tilt:
      raise ValueError(f"gripper would be tilted {tilt:.0f} deg at the drop (max {max_tilt:.0f})")
    memo["lower"] = plan
    return plan

  def rise(q, raw):
    lw = memo["lower"]
    return ReplayPlan(lw.t, lw.q[::-1].copy(), np.full(len(lw.t), grip_open), lw.pos_err[::-1].copy(),
                      lw.point_err_deg[::-1].copy(), lw.targets[::-1].copy())

  home = home_q6()
  return [
    ("lift", check[0][1]),
    ("carry", PolicyTarget(np.array([drop[0], drop[1], carry_z]))),
    ("lower", Deferred(lower)),
    ("open", Deferred(lambda q, raw: joint_plan(kin, q, q, 0.8, grip_hold, grip_open))),
    ("rise", Deferred(rise)),
    ("home", Deferred(lambda q, raw: joint_plan(kin, q, home, 3.0, grip_open, jmap.to_ticks("gripper", home[5])))),
  ]


def container_sequence(kin: ArmKinematics, jmap: JointMap, cont, q6: np.ndarray, grip_hold: float, grip_open: float,
                       obj_radius: float, site_lo: np.ndarray, site_hi: np.ndarray, carry_clear: float = 0.06,
                       drop_clear: float = 0.03, wall_margin: float = 0.015, v_vert: float = 0.03,
                       v_move: float = 0.06, max_tilt: float = 70.0, max_ik_err: float = 0.003) -> list[tuple[str, ReplayPlan]]:
  """Carry a held object into a container (containers.Container) and let go:
  lift (to `carry_clear` above the rim for the object's bottom) -> straight move over the drop
  point (inside the footprint, nearest the arm, object `wall_margin` clear of the walls) ->
  lower (object bottom `drop_clear` above the rim) -> open -> rise -> home.
  Raises ValueError (nothing moves) if any part is unreachable, leaves the site limits, or
  needs the gripper tilted more than `max_tilt` from vertical at the drop."""
  tcp, _ = kin.fk(q6)
  drop, carry_z, drop_z = drop_geometry(kin, cont, tcp, obj_radius, site_lo, site_hi, carry_clear, drop_clear, wall_margin)
  seq = []
  q = q6
  for name, end, v, grip in (("lift", np.array([tcp[0], tcp[1], carry_z]), v_vert, grip_hold),
                             ("carry", np.array([drop[0], drop[1], carry_z]), v_move, grip_hold),
                             ("lower", np.array([drop[0], drop[1], drop_z]), v_vert, grip_hold)):
    start, _ = kin.fk(q)
    plan = line_plan(kin, q, end, max(0.6, np.linalg.norm(end - start) / v), grip)
    seq.append((name, plan))
    q = plan.q[-1]
  seq.append(("open", joint_plan(kin, q, q, 0.8, grip_hold, grip_open)))
  lower = seq[-2][1]  # Rise = the lowering path backwards (smooth, known clear of the walls).
  up = ReplayPlan(lower.t, lower.q[::-1].copy(), np.full(len(lower.t), grip_open), lower.pos_err[::-1].copy(),
                  lower.point_err_deg[::-1].copy(), lower.targets[::-1].copy())
  seq.append(("rise", up))
  home = home_q6()
  seq.append(("home", joint_plan(kin, up.q[-1], home, 3.0, grip_open, jmap.to_ticks("gripper", home[5]))))
  for name, plan in seq:
    if plan.pos_err.max() > max_ik_err:
      raise ValueError(f"{name}: out of reach (IK error {plan.pos_err.max() * 1e3:.0f} mm)")
    if name != "home" and (np.any(plan.targets[:, :2] < site_lo[:2]) or np.any(plan.targets[:, :2] > site_hi[:2])):
      raise ValueError(f"{name}: leaves the site limits")
  tilt = tilt_deg(kin, seq[2][1].q[-1])
  if tilt > max_tilt:
    raise ValueError(f"gripper would be tilted {tilt:.0f} deg from vertical at the drop (max {max_tilt:.0f})")
  return seq


class GraspReplaySession:
  """Window session: FOLLOW (hover above the orange) -> G -> GRASP (replay) -> HOLDING ->
  P -> PLACE (lower, open, rise, home) -> FOLLOW. H: go home (from FOLLOW / HOLDING).
  `auto`: start the grasp by itself once the hover has settled (< settle_tol for settle_s).
  `speed` scales every played motion (grasp, place, home) vs the demo / default timing.

  Trials (`trials` > 0, implies auto): grasp -> hold `hold_s` -> judge -> place + home -> wait
  for SPACE (move the orange first) -> next trial. A grasp counts as picked when the gripper
  stopped more than `held_gap` ticks short of its closing goal (an empty gripper closes fully).
  Y / N while waiting override the last verdict. Every trial is appended to `log_path` (JSON).

  `container` (containers.Container): P carries the object into it and lets go
  (container_sequence; if that is not possible the object is put back where it was picked).
  D always puts it back down where it was picked.
  """

  def __init__(self, core, tracker, demo: GraspDemo, fcal: dict[str, MotorCalib], speed: float = 1.0, auto: bool = False,
               settle_tol: float = 0.008, settle_s: float = 1.5, max_ik_err: float = 0.003, table_margin: float = 0.01,
               trials: int = 0, hold_s: float = 2.0, held_gap: int = 40, log_path: Path | None = None, container=None):
    self.core, self.tr, self.demo, self.fcal = core, tracker, demo, fcal
    self.container = container
    self.obj_radius = 0.026
    self.jmap = JointMap.load()
    self.kin = ArmKinematics()
    self.speed, self.auto = speed, auto
    self.settle_tol, self.settle_s = settle_tol, settle_s
    self.max_ik_err, self.table_margin = max_ik_err, table_margin
    self.state = "FOLLOW"
    self.queue: list[tuple[str, ReplayPlan]] = []
    self.player: PlanPlayer | None = None
    self.phase = ""
    self.msg = ""
    self.orange: np.ndarray | None = None
    self._settled_ticks = 0
    self.grasps = 0
    self.player_gap = 0
    self.armed = auto or trials > 0  # grasp by itself once settled
    self.trials, self.hold_s, self.held_gap, self.log_path = trials, hold_s, held_gap, log_path
    self.results: list[dict] = []
    self.trial: dict | None = None
    self._t_hold = 0.0
    self._t_trial = time.time()
    if trials:
      self.msg = f"trial 1/{trials}: going to the {tracker.name}"

  @property
  def bus(self):
    return self.core.plant.bus

  def _q6(self) -> tuple[np.ndarray, dict[str, int]]:
    raw = self.bus.sync_read("Present_Position")
    return np.array([self.jmap.to_rad(n, raw[n]) for n in JOINT_ORDER]), raw

  def _start(self, state: str, seq: list[tuple[str, ReplayPlan]]) -> None:
    if self.core.following:
      self.core.toggle()
    self.state, self.queue, self.player = state, list(seq), None

  def press_g(self) -> None:
    if self.state != "FOLLOW":
      return
    self.armed = False
    if self.tr.center is None or time.time() - self.tr.last_seen > 0.5:
      self.msg = f"no fresh {self.tr.name} position - cannot grasp"
      return
    orange = self.tr.center.copy()
    q6, _ = self._q6()
    plan = retarget(self.demo, orange, q_start=q6, kin=self.kin)
    table_z = orange[2] - self.tr.radius
    lowest = plan.targets[:, 2].min()
    refused = None
    if plan.pos_err.max() > self.max_ik_err:
      refused = f"REFUSED: grasp path not reachable here (IK error {plan.pos_err.max() * 1e3:.1f} mm)"
    elif lowest < table_z + self.table_margin:
      refused = f"REFUSED: grasp path goes to {(lowest - table_z) * 100:.1f} cm above the table"
    elif np.any(plan.targets[:, :2] > self.core.targets.hi[:2]) or np.any(plan.targets[:, :2] < self.core.targets.lo[:2]):
      refused = "REFUSED: grasp path leaves the site limits (configs/real_workspace.json, e.g. the wall)"
    if self.trials:
      self.trial = {"trial": len(self.results) + 1, "orange": orange.round(4).tolist(), "radius": round(float(self.tr.radius), 4),
                    "seen_by": self.tr.seen_by, "hover_s": round(time.time() - self._t_trial, 2), "speed": self.speed}
    if refused:
      self.msg = refused
      print(f"[replay] {refused}", flush=True)
      if self.trial is not None:
        self._record("refused", False)
      return
    self.orange = orange
    self.obj_radius = float(self.tr.radius)
    self.msg = f"grasping {self.tr.name} at ({orange[0]:.3f}, {orange[1]:.3f}, {orange[2]:.3f}); lowest TCP {(lowest - table_z) * 100:.1f} cm above table"
    print(f"[replay] {self.msg}", flush=True)
    self._start("GRASP", [("grasp", plan)])

  def press_p(self) -> None:
    """Into the container if there is one (else / if impossible: back where it was picked)."""
    if self.state != "HOLDING":
      return
    if self.container is None:
      self.press_d()
      return
    q6, raw = self._q6()
    try:
      seq = container_steps(self.kin, self.jmap, self.container, q6, raw["gripper"] - 30, float(self.demo.gripper_goal.max()),
                            self.obj_radius, self.core.targets.lo, self.core.targets.hi)
    except ValueError as e:
      self.msg = f"cannot place into the {self.container.name}: {e} - putting it back"
      print(f"[replay] {self.msg}", flush=True)
      self.press_d()
      return
    if self.trial is not None:
      self.trial["placed_in"] = self.container.name
    self.msg = f"placing into the {self.container.name}"
    self._start("PLACE", seq)

  def press_d(self) -> None:
    """Put the held object back down where it was picked, open, rise, home."""
    if self.state != "HOLDING":
      return
    q6, raw = self._q6()
    oz = self.orange[2] if self.orange is not None else self.demo.orange[2]
    if self.trial is not None:
      self.trial["placed_in"] = "table"
    self._start("PLACE", place_sequence(self.kin, self.jmap, self.demo, q6, raw["gripper"] - 30, oz))

  def press_h(self) -> None:
    if self.state not in ("FOLLOW", "HOLDING"):
      return
    q6, raw = self._q6()
    home = home_q6()
    grip = raw["gripper"] - 30 if self.state == "HOLDING" else self.jmap.to_ticks("gripper", home[5])
    nxt = "HOLDING" if self.state == "HOLDING" else "FOLLOW"
    self._start("HOME", [("home", joint_plan(self.kin, q6, home, 3.0, raw["gripper"], grip))])
    self._after_home = nxt

  def step(self) -> None:
    if self.state == "FOLLOW":
      self.core.tick(self.tr.hover_target() if self.core.following else None)
      err = np.linalg.norm(self.core.tcp - self.core.target)
      fresh = self.tr.center is not None and time.time() - self.tr.last_seen < 0.5
      ok = self.core.following and fresh and err < self.settle_tol
      self._settled_ticks = self._settled_ticks + 1 if ok else 0
      if self.armed and self._settled_ticks * self.core.plant.dt >= self.settle_s:
        self.press_g()
      return
    if self.state == "HOLDING" and self.trials and self.trial is not None and time.time() - self._t_hold >= self.hold_s:
      if self.trial.get("picked"):
        self.press_p()
      else:  # Nothing in the gripper: straight home.
        q6, raw = self._q6()
        home = home_q6()
        self._start("PLACE", [("home", joint_plan(self.kin, q6, home, 3.0, raw["gripper"], self.jmap.to_ticks("gripper", home[5])))])
      return
    if self.state not in ("GRASP", "PLACE", "HOME"):
      return
    if self.player is None:
      if not self.queue:
        self._finish()
        return
      self.phase, item = self.queue.pop(0)
      if isinstance(item, PolicyTarget):
        self.player = PolicyMove(self.core, item.target)
        self.msg = f"{self.phase}: RL reach policy to ({item.target[0]:.3f}, {item.target[1]:+.3f}, {item.target[2]:.3f})"
      else:
        if isinstance(item, Deferred):
          try:
            q6, raw = self._q6()
            item = item.build(q6, raw)
          except ValueError as e:
            self.msg = f"{self.phase} not possible: {e} - holding (D = put it back, H = home)"
            print(f"[replay] {self.msg}", flush=True)
            self.queue, self.state = [], "HOLDING"
            if self.trial is not None:
              self._record("stopped", self.trial.get("picked", False))
            self._resync()
            return
        grip_torque = 350 if self.phase in ("grasp",) else 250
        self.player = PlanPlayer(self.bus, item, self.jmap, self.fcal, speed=self.speed,
                                 gripper_torque=grip_torque, approach_s=(1.5 if self.phase == "grasp" else 0.3) / min(self.speed, 1.0))
    self.player.step()
    if self.player.done:
      st = self.player.status
      self.player_gap = self.player.gripper_gap
      print(f"[replay] {self.phase}: {st}", flush=True)
      self.player = None
      if self.phase == "grasp" and self.trial is not None:
        self.trial.update(grasp_status=st, gripper_gap=self.player_gap)
      if self.phase == "lower" and self.trial is not None and self.trial.get("placed_in") not in (None, "table"):
        self.trial["held_at_drop"] = self.player_gap > 15  # still stopped short by the object above the box
      if st != "ok":
        self.msg = f"{self.phase} {st} - holding"
        self.queue = []
        if self.trial is not None:
          self._record("stopped", False)
        self.state = "HOLDING" if self.state == "GRASP" else "FOLLOW"
        self._resync()

  def _resync(self) -> None:
    """Hand control back to the follow core without a jump (its policy restarts at the present pose)."""
    q = self.core.plant.read_q()
    self.core.ctrl.reset(q)
    self.core.q, self.core.tcp = q, self.core.ctrl.kin.tcp(q)
    self.core.target = self.core.tcp.copy()

  def _finish(self) -> None:
    if self.state == "GRASP":
      self.grasps += 1
      where = f"into the {self.container.name}" if self.container is not None else "down"
      self.state, self.msg = "HOLDING", f"grasp done - holding the {self.tr.name} (P = put it {where}, D = put it back, H = home)"
      self._t_hold = time.time()
      if self.trial is not None:
        held = self.trial.get("gripper_gap", 0) > self.held_gap
        self.trial["picked"] = held
        self.msg = f"trial {self.trial['trial']}: {'PICKED' if held else 'MISSED'} (gripper gap {self.trial.get('gripper_gap')} ticks) - placing"
    elif self.state == "PLACE":
      self.state, self.msg = "FOLLOW", f"{self.tr.name} placed, arm home (F = follow, G = grasp again)"
      if self.trial is not None:
        self._record("ok", self.trial["picked"])
    else:
      self.state = getattr(self, "_after_home", "FOLLOW")
      self.msg = "home"
    self._resync()

  def _record(self, status: str, picked: bool) -> None:
    t = self.trial
    t.update(status=status, picked=picked, total_s=round(time.time() - self._t_trial, 2))
    self.results.append(t)
    self.trial = None
    n_ok = sum(r["picked"] for r in self.results)
    placed = f", placed in {t['placed_in']}" + ("" if t.get("held_at_drop", True) else " (DROPPED before the box)") if t.get("placed_in") else ""
    print(f"[trial] {t['trial']}/{self.trials} {'PICKED' if picked else 'MISSED'} ({status}) {self.tr.name} "
          f"({t['orange'][0]:.3f}, {t['orange'][1]:.3f}) gap {t.get('gripper_gap')}{placed} - {n_ok}/{len(self.results)} so far", flush=True)
    self._save()
    if len(self.results) >= self.trials:
      self.msg = f"ALL {self.trials} TRIALS DONE: {n_ok}/{self.trials} picked (Y/N fixes the last one, Esc quits)"
    else:
      self.msg = f"trial {t['trial']} {'PICKED' if picked else 'MISSED'} - put a new {self.tr.name} out, then SPACE (Y/N fixes this verdict)"

  def _save(self) -> None:
    if self.log_path is None:
      return
    self.log_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = sum(r["picked"] for r in self.results)
    self.log_path.write_text(json.dumps({"demo": self.demo.meta.get("created"), "method": self.demo.meta.get("method", "orange"),
                                         "object": self.tr.name, "place": self.container.name if self.container else None,
                                         "trials": self.results,
                                         "picked": n_ok, "done": len(self.results)}, indent=2) + "\n")

  def press_space(self) -> None:
    """Next trial (after moving the orange)."""
    if not self.trials or self.state != "FOLLOW" or self.trial is not None or len(self.results) >= self.trials:
      return
    self.armed, self._t_trial = True, time.time()
    self.msg = f"trial {len(self.results) + 1}/{self.trials}: going to the {self.tr.name}"
    if not self.core.following:
      self.core.toggle()

  def mark(self, picked: bool) -> None:
    """Y / N: correct the last trial's verdict."""
    if self.results and self.trial is None:
      self.results[-1]["picked"] = picked
      self.results[-1]["verdict_by"] = "user"
      self._save()
      n_ok = sum(r["picked"] for r in self.results)
      self.msg = f"trial {self.results[-1]['trial']} marked {'PICKED' if picked else 'MISSED'} ({n_ok}/{len(self.results)})"
      print(f"[trial] {self.results[-1]['trial']} corrected by user: {'PICKED' if picked else 'MISSED'}", flush=True)

  def status(self) -> str:
    if self.state in ("GRASP", "PLACE", "HOME") and self.player is not None:
      return f"{self.state}: {self.phase} {self.player.progress * 100:3.0f}%  {self.msg}"
    if self.state == "FOLLOW":
      ready = "SETTLED - G = grasp" if self._settled_ticks else "G = grasp when settled above the orange"
      mode = f"TRIALS {sum(r['picked'] for r in self.results)}/{len(self.results)}" if self.trials else ("auto" if self.auto else "manual")
      return f"REPLAY ({mode}): {ready} | F follow, H home | {self.msg}"
    return f"{self.state}: {self.msg}"
