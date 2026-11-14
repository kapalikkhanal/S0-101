"""Conventional IK baselines.

1. ``DlsIkController``: batched damped-least-squares (DLS) resolved-rate IK (closed loop).
2. ``solve_ik_batch`` + ``JointInterpolationController``: global multi-restart IK solved
   once per target, then a synchronized trapezoidal joint-space move (open loop in joint
   space, exact in Cartesian space up to servo tracking).

DLS details:

Runs on the GPU inside the same MjLab environment and emits actions through the *same*
``IncrementalJointPositionAction`` as the RL policy, so both controllers share the
per-step joint-increment limit, soft joint limits, max-lead clamp, servos and physics:

  e      = target - tcp                              (base frame)
  dx     = gain * e, norm-limited to v_max * dt
  dq     = J^T (J J^T + lambda^2 I)^-1 dx  +  (I - J^+ J) k_null (q_home - q)
  action = clip(dq / action_scale, -1, 1)

J is the geometric position Jacobian of the TCP w.r.t. the four arm joints, built from
the joint axes/anchors MuJoCo computes for each world (J_i = a_i x (p_tcp - p_i)).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from so101_reach.sim import so101

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass
class DlsIkCfg:
  gain: float = 0.35
  """Fraction of the Cartesian error commanded per 20 ms step."""
  v_max: float = 0.5
  """Cartesian speed limit [m/s]."""
  damping: float = 0.02
  """DLS damping lambda [m]."""
  null_gain: float = 0.02
  """Null-space pull toward the home posture per step."""


class DlsIkController:
  def __init__(self, env: ManagerBasedRlEnv, cfg: DlsIkCfg | None = None, command_name: str = "ee_target", action_name: str = "arm"):
    self.env = env
    self.cfg = cfg or DlsIkCfg()
    self.cmd = env.command_manager.get_term(command_name)
    self.act = env.action_manager.get_term(action_name)
    robot = env.scene["robot"]
    ids, _ = robot.find_joints(so101.ARM_JOINTS, preserve_order=True)
    self.local_ids = torch.tensor(ids, device=env.device)
    self.global_jnt = robot.indexing.joint_ids[self.local_ids].long()
    self.robot = robot
    self.q_home = torch.tensor([so101.HOME_JOINT_POS[n] for n in so101.ARM_JOINTS], device=env.device)
    self.dt = env.step_dt

  def jacobian(self) -> torch.Tensor:
    """(N, 3, 4) TCP position Jacobian in the base frame (base frame == world rotation)."""
    d = self.env.sim.data
    axis = d.xaxis[:, self.global_jnt]  # (N, 4, 3)
    anchor = d.xanchor[:, self.global_jnt]
    tcp_w = self.robot.data.site_pos_w[:, self.cmd.tcp_site_id]
    return torch.cross(axis, tcp_w[:, None, :] - anchor, dim=-1).transpose(1, 2)

  def __call__(self, obs=None) -> torch.Tensor:
    del obs
    c = self.cfg
    e = self.cmd.error_b
    dx = c.gain * e
    n = torch.norm(dx, dim=-1, keepdim=True)
    dx = dx * torch.clamp(c.v_max * self.dt / n.clamp(min=1e-9), max=1.0)
    J = self.jacobian()
    JJt = J @ J.transpose(1, 2) + (c.damping**2) * torch.eye(3, device=J.device)
    J_pinv = J.transpose(1, 2) @ torch.linalg.inv(JJt)  # (N, 4, 3)
    dq = (J_pinv @ dx.unsqueeze(-1)).squeeze(-1)
    q = self.robot.data.joint_pos[:, self.local_ids]
    null = torch.eye(4, device=J.device) - J_pinv @ J
    dq = dq + (null @ (c.null_gain * (self.q_home - q)).unsqueeze(-1)).squeeze(-1)
    return torch.clamp(dq / self.act.cfg.scale, -1.0, 1.0)


# Global IK + joint-space interpolation baseline.

_IK: dict = {}


def _ik_init(clearance: float) -> None:
  import mujoco

  from so101_reach.sim.workspace import soft_joint_limits

  model = so101.make_cpu_model()
  coll = (model.geom_contype > 0) | (model.geom_conaffinity > 0)
  model.geom_margin[coll] = clearance
  _IK.update(
    model=model,
    data=mujoco.MjData(model),
    qadr=np.array([model.jnt_qposadr[model.joint(n).id] for n in so101.ARM_JOINTS]),
    dadr=np.array([model.jnt_dofadr[model.joint(n).id] for n in so101.ARM_JOINTS]),
    fadr=np.array([model.jnt_qposadr[model.joint(n).id] for n in so101.FIXED_JOINTS]),
    fixed=np.array([so101.FIXED_JOINT_TARGETS[n] for n in so101.FIXED_JOINTS]),
    site=model.site(so101.TCP_SITE).id,
    lim=soft_joint_limits(model),
    clearance=clearance,
  )


def _ik_one(args) -> tuple[np.ndarray, float]:
  import mujoco

  target, q_start, n_restarts, iters, seed = args
  m, d = _IK["model"], _IK["data"]
  lim, qadr, dadr = _IK["lim"], _IK["qadr"], _IK["dadr"]
  rng = np.random.default_rng(seed)
  seeds = [np.clip(q_start, lim[:, 0], lim[:, 1])] + [rng.uniform(lim[:, 0], lim[:, 1]) for _ in range(n_restarts - 1)]
  jacp = np.zeros((3, m.nv))
  best, best_key = None, (np.inf, np.inf)
  for q in seeds:
    q = q.copy()
    d.qpos[_IK["fadr"]] = _IK["fixed"]
    for _ in range(iters):
      d.qpos[qadr] = q
      mujoco.mj_kinematics(m, d)
      mujoco.mj_comPos(m, d)
      e = target - d.site_xpos[_IK["site"]]
      if np.linalg.norm(e) < 1e-4:
        break
      mujoco.mj_jacSite(m, d, jacp, None, _IK["site"])
      J = jacp[:, dadr]
      q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), e), lim[:, 0], lim[:, 1])
    d.qpos[qadr] = q
    mujoco.mj_kinematics(m, d)
    err = float(np.linalg.norm(target - d.site_xpos[_IK["site"]]))
    mujoco.mj_collision(m, d)
    free = d.ncon == 0 or bool(np.all(d.contact.dist[: d.ncon] >= _IK["clearance"]))
    # Prefer: converged (< 1 mm) and collision-free, then the least joint motion.
    key = (0 if (err < 1e-3 and free) else 1, np.abs(q - q_start).max() if err < 1e-3 else err)
    if key < best_key:
      best, best_key = q, key
  return best, best_key[1] if best_key[0] == 1 else 0.0


def solve_ik_batch(
  targets: np.ndarray, q_start: np.ndarray, n_restarts: int = 8, iters: int = 80, clearance: float = 0.002, workers: int | None = None
) -> np.ndarray:
  """Multi-restart, joint-limit-aware, collision-checked IK for many targets (CPU MuJoCo)."""
  import multiprocessing as mp

  workers = workers or max(1, mp.cpu_count() - 2)
  jobs = [(targets[i].astype(np.float64), q_start[i].astype(np.float64), n_restarts, iters, i) for i in range(len(targets))]
  with mp.get_context("spawn").Pool(workers, initializer=_ik_init, initargs=(clearance,)) as pool:
    res = pool.map(_ik_one, jobs, chunksize=16)
  return np.stack([r[0] for r in res]).astype(np.float32)


class JointInterpolationController:
  """Stream a synchronized straight-line joint trajectory (trapezoidal speed profile) from the
  current commanded target to an IK solution, using the same per-step increment limit as the
  policy. The trajectory is generated from an internal reference, so increments still in
  flight (action delay) are not re-issued; a weak feedback term absorbs limit clamping."""

  def __init__(
    self, env: ManagerBasedRlEnv, q_goal: torch.Tensor, accel: float = 10.0, feedback: float = 0.2, action_name: str = "arm"
  ):
    self.act = env.action_manager.get_term(action_name)
    self.dt = env.step_dt
    self.accel = accel
    self.feedback = feedback
    self.q_goal = q_goal.to(env.device)
    self.q_ref: torch.Tensor | None = None
    self.step_size = torch.zeros(env.num_envs, device=env.device)

  def set_goal(self, q_goal: torch.Tensor) -> None:
    self.q_goal = q_goal.to(self.q_goal.device)
    self.q_ref = None
    self.step_size.zero_()

  def __call__(self, obs=None) -> torch.Tensor:
    del obs
    scale = self.act.cfg.scale
    if self.q_ref is None or self.q_ref.shape != self.act.joint_target.shape:
      self.q_ref = self.act.joint_target.clone()
    diff = self.q_goal - self.q_ref
    r = diff.abs().max(dim=-1).values
    a_step = self.accel * self.dt * self.dt  # increment change per step [rad]
    s = torch.minimum(self.step_size + a_step, torch.full_like(r, scale))
    s = torch.minimum(s, torch.sqrt(2.0 * a_step * r))
    s = torch.minimum(s, r)
    self.step_size = s
    planned = diff * (s / r.clamp(min=1e-9)).unsqueeze(-1)
    self.q_ref = self.q_ref + planned
    correction = self.feedback * (self.q_ref - self.act.joint_target - planned)
    return torch.clamp((planned + correction) / scale, -1.0, 1.0)
