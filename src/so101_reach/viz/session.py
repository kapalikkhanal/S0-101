"""Interactive reaching session shared by the GUI viewers and the video recorder.

Wraps a single-env MjLab evaluation environment plus a controller and exposes
``command(xyz)``, ``random_target()``, ``home()`` and a live ``status()`` snapshot
(target, TCP, error, joints, elapsed time, MOVING / REACHED / FAILED).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from so101_reach.sim import so101
from so101_reach.sim.workspace import load_bank

Controller = Literal["native", "nvidia", "ik", "ik_joint"]


def make_controller(kind: Controller, env, checkpoint: Path | None = None):
  """Return a callable obs -> action for the given controller type."""
  from so101_reach.checkpoints import policy_io
  from so101_reach.evaluation.ik_controller import DlsIkController, JointInterpolationController

  if kind == "native":
    return policy_io.load_native_policy(env, checkpoint or policy_io.latest_checkpoint())
  if kind == "nvidia":
    return policy_io.load_nvidia_obs_policy(str(env.device))
  if kind == "ik":
    return DlsIkController(env)
  if kind == "ik_joint":
    act = env.action_manager.get_term("arm")
    return JointInterpolationController(env, act.joint_target.clone())
  raise ValueError(kind)


def solve_ik_single(target: np.ndarray, q_start: np.ndarray) -> np.ndarray:
  from so101_reach.evaluation import ik_controller as ikc

  if not ikc._IK:
    ikc._ik_init(0.002)
  q, _ = ikc._ik_one((target.astype(np.float64), q_start.astype(np.float64), 8, 80, 0))
  return q.astype(np.float32)


@dataclass
class ReachStatus:
  target: np.ndarray
  tcp: np.ndarray
  error_mm: float
  speed_mm_s: float
  joints_deg: np.ndarray
  elapsed_s: float
  state: str
  reach_time_s: float | None
  feasibility: str

  def lines(self) -> list[tuple[str, str]]:
    j = self.joints_deg
    return [
      ("State", self.state + (f" in {self.reach_time_s:.2f} s" if self.reach_time_s is not None else "")),
      ("Target [m]", f"x={self.target[0]:+.3f} y={self.target[1]:+.3f} z={self.target[2]:+.3f}"),
      ("TCP [m]", f"x={self.tcp[0]:+.3f} y={self.tcp[1]:+.3f} z={self.tcp[2]:+.3f}"),
      ("Error", f"{self.error_mm:.1f} mm"),
      ("TCP speed", f"{self.speed_mm_s:.0f} mm/s"),
      ("Elapsed", f"{self.elapsed_s:.2f} s"),
      ("Joints [deg]", f"pan {j[0]:+.1f} lift {j[1]:+.1f} elbow {j[2]:+.1f} wrist {j[3]:+.1f}"),
      ("Workspace", self.feasibility),
    ]


class ReachSession:
  def __init__(self, env, controller_kind: Controller, policy, tol_mm: float = 10.0, hold_s: float = 0.3, timeout_s: float = 6.0):
    self.env = env
    self.kind = controller_kind
    self.policy = policy
    self.cmd = env.command_manager.get_term("ee_target")
    self.act = env.action_manager.get_term("arm")
    self.robot = env.scene["robot"]
    ids, _ = self.robot.find_joints(so101.ARM_JOINTS, preserve_order=True)
    self.arm_ids = torch.tensor(ids, device=env.device)
    st = self.cmd.stage
    st.success_tol = tol_mm / 1e3
    st.hold_time_s = hold_s
    self.timeout_s = timeout_s
    bank = load_bank("eval")
    self.bank_tcp = torch.tensor(bank.tcp_pos, device=env.device)
    self.rng = np.random.default_rng()
    self.t_cmd_step = 0
    self.feasibility = "-"
    self.wall_t0 = time.perf_counter()

  @property
  def step_count(self) -> int:
    return int(self.env.common_step_counter)

  def feasibility_check(self, xyz: np.ndarray) -> tuple[bool, str]:
    d = torch.norm(self.bank_tcp - torch.tensor(xyz, device=self.env.device, dtype=torch.float32), dim=-1).min().item()
    if d <= 0.01:
      return True, f"feasible (verified point {d * 1e3:.1f} mm away)"
    return False, f"OUTSIDE verified workspace: nearest reachable point {d * 1e3:.0f} mm away"

  def command(self, xyz) -> str:
    xyz = np.asarray(xyz, dtype=np.float32)
    ok, msg = self.feasibility_check(xyz)
    self.feasibility = msg
    self.cmd.set_target(torch.tensor(xyz, device=self.env.device)[None])
    self.t_cmd_step = self.step_count
    if self.kind == "ik_joint":
      q0 = self.robot.data.joint_pos[0, self.arm_ids].cpu().numpy()
      self.policy.set_goal(torch.tensor(solve_ik_single(xyz, q0))[None])
    return msg

  def project_to_workspace(self, xyz) -> tuple[np.ndarray, float]:
    """Nearest verified-feasible point if `xyz` is more than 1 cm outside the workspace."""
    p = torch.as_tensor(np.asarray(xyz, dtype=np.float32), device=self.env.device)
    d = torch.norm(self.bank_tcp - p, dim=-1)
    i = int(d.argmin())
    dist = float(d[i])
    return (self.bank_tcp[i].cpu().numpy() if dist > 0.01 else np.asarray(xyz, dtype=np.float32)), dist

  def track(self, xyz, min_move: float = 0.002) -> np.ndarray:
    """Continuously moving target (e.g. mouse): projected onto the workspace and re-commanded
    only when it moves by more than `min_move`, so status/timers stay meaningful."""
    proj, dist = self.project_to_workspace(xyz)
    cur = self.cmd.target_pos_b[0].cpu().numpy()
    if np.linalg.norm(proj - cur) > min_move:
      self.cmd.set_target(torch.tensor(proj, device=self.env.device)[None])
      self.t_cmd_step = self.step_count
      self.feasibility = "feasible" if dist <= 0.01 else f"cursor outside workspace -> snapped ({dist * 1e3:.0f} mm)"
      if self.kind == "ik_joint":
        q0 = self.robot.data.joint_pos[0, self.arm_ids].cpu().numpy()
        self.policy.set_goal(torch.tensor(solve_ik_single(proj, q0))[None])
    return proj

  def random_target(self) -> np.ndarray:
    xyz = self.bank_tcp[int(self.rng.integers(len(self.bank_tcp)))].cpu().numpy()
    self.command(xyz)
    return xyz

  def home(self) -> None:
    from so101_reach.tasks.mdp.events import fk_tcp_cpu

    self.command(fk_tcp_cpu({**so101.HOME_JOINT_POS}))

  def reached(self) -> bool:
    return bool(self.cmd.ever_succeeded[0])

  def status(self) -> ReachStatus:
    elapsed = (self.step_count - self.t_cmd_step) * self.env.step_dt
    err = float(self.cmd.distance[0]) * 1e3
    if self.reached():
      state, rt = "REACHED", float(self.cmd.success_time_s[0])
    elif elapsed > self.timeout_s:
      state, rt = "FAILED (timeout)", None
    else:
      state, rt = "MOVING", None
    return ReachStatus(
      target=self.cmd.target_pos_b[0].cpu().numpy(),
      tcp=self.cmd.tcp_pos_b[0].cpu().numpy(),
      error_mm=err,
      speed_mm_s=float(torch.norm(self.cmd.tcp_lin_vel_b[0])) * 1e3,
      joints_deg=np.degrees(self.robot.data.joint_pos[0, self.arm_ids].cpu().numpy()),
      elapsed_s=elapsed,
      state=state,
      reach_time_s=rt,
      feasibility=self.feasibility,
    )


def make_session(
  controller: Controller = "native",
  checkpoint: Path | None = None,
  device: str = "cuda:0",
  tol_mm: float = 10.0,
  render: bool = False,
  width: int = 1280,
  height: int = 720,
) -> ReachSession:
  from mjlab.envs import ManagerBasedRlEnv

  from so101_reach.tasks.eval_env_cfg import make_eval_env_cfg

  cfg = make_eval_env_cfg("nvidia" if controller == "nvidia" else "native", num_envs=1)
  cfg.viewer.width, cfg.viewer.height = width, height
  env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode="rgb_array" if render else None)
  env.reset()
  policy = make_controller(controller, env, checkpoint)
  if controller in ("native",):
    env.reset()  # The RSL-RL wrapper resets on construction; start clean at home.
  sess = ReachSession(env, controller, policy, tol_mm=tol_mm)
  sess.home()
  return sess
