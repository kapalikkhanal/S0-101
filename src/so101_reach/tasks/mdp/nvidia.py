"""Observation/action terms reproducing the nvidia/so101-reach interface inside MjLab."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch

from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import Entity
from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

  from so101_reach.tasks.mdp.commands import ReachCommand

NV_OBS_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
NV_ACT_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")


@dataclass(kw_only=True)
class NvidiaDeltaActionCfg(BaseActionCfg):
  """target = q_measured + clip(a, -1, 1) * scale -> clamp(limits) -> clamp(q +- max_step).

  rebase:
    "policy_step": target formed once per 50 Hz policy step (model-card deployment contract).
    "every_10ms":  target re-based on the measured position every 10 ms, matching the
                   Isaac Lab training action term (applied each 100 Hz physics substep).
  """

  actuator_names: tuple[str, ...] = NV_ACT_JOINTS
  scale: float = 0.2
  max_step: float = 0.10
  joint_limits: dict[str, tuple[float, float]] = field(default_factory=dict)
  fixed_targets: dict[str, float] = field(default_factory=lambda: {"wrist_roll": 0.0, "gripper": 0.0})
  rebase: Literal["policy_step", "every_10ms"] = "policy_step"
  physics_dt: float = 0.005
  preserve_order: bool = True

  def __post_init__(self):
    self.transmission_type = TransmissionType.JOINT

  def build(self, env: ManagerBasedRlEnv) -> NvidiaDeltaAction:
    return NvidiaDeltaAction(self, env)


class NvidiaDeltaAction(BaseAction):
  cfg: NvidiaDeltaActionCfg

  def __init__(self, cfg: NvidiaDeltaActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    n = self.num_envs
    lo = [cfg.joint_limits[nm][0] for nm in self._target_names]
    hi = [cfg.joint_limits[nm][1] for nm in self._target_names]
    self._lo = torch.tensor(lo, device=self.device).expand(n, -1)
    self._hi = torch.tensor(hi, device=self.device).expand(n, -1)
    self._clipped = torch.zeros(n, self.action_dim, device=self.device)
    self._target = torch.zeros(n, self.action_dim, device=self.device)
    fixed_ids, fixed_names = self._entity.find_joints(tuple(cfg.fixed_targets), preserve_order=True)
    self._fixed_ids = torch.tensor(fixed_ids, device=self.device, dtype=torch.long)
    self._fixed_val = torch.tensor([cfg.fixed_targets[k] for k in fixed_names], device=self.device).expand(n, -1)
    self._substep = 0
    self._rebase_every = max(1, round(0.01 / cfg.physics_dt))

  @property
  def clipped_action(self) -> torch.Tensor:
    return self._clipped

  @property
  def joint_target(self) -> torch.Tensor:
    return self._target

  def _compute_target(self) -> None:
    q = self._entity.data.joint_pos[:, self._target_ids]
    tgt = torch.clamp(q + self._clipped * self.cfg.scale, self._lo, self._hi)
    self._target[:] = torch.clamp(tgt, q - self.cfg.max_step, q + self.cfg.max_step)

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions
    self._clipped[:] = torch.nan_to_num(actions, nan=0.0).clamp(-1.0, 1.0)
    self._substep = 0
    self._compute_target()
    self._processed_actions = self._target

  def apply_actions(self) -> None:
    if self.cfg.rebase == "every_10ms" and self._substep > 0 and self._substep % self._rebase_every == 0:
      self._compute_target()
    self._substep += 1
    self._entity.set_joint_position_target(self._target, joint_ids=self._target_ids)
    self._entity.set_joint_position_target(self._fixed_val, joint_ids=self._fixed_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    self._raw_actions[ids] = 0.0
    self._clipped[ids] = 0.0
    self._target[ids] = self._entity.data.joint_pos[ids][:, self._target_ids]

  def sync_target_to_measured(self, env_ids: torch.Tensor | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    self._target[ids] = self._entity.data.joint_pos[ids][:, self._target_ids]


def nv_joint_pos_rel(env: ManagerBasedRlEnv, offsets: tuple[float, ...]) -> torch.Tensor:
  """Isaac joint_pos_rel over the six joints in descriptor order; slots [4:6] zeroed."""
  robot: Entity = env.scene["robot"]
  ids, _ = robot.find_joints(NV_OBS_JOINTS, preserve_order=True)
  q = robot.data.joint_pos[:, ids] - torch.tensor(offsets, device=env.device)
  q[:, 4:6] = 0.0
  return q


def nv_ee_position(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  cmd: ReachCommand = env.command_manager.get_term(command_name)  # type: ignore[assignment]
  return cmd.tcp_pos_b


def nv_goal_pose(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Goal as [x, y, z, qw, qx, qy, qz] with the identity quaternion (WXYZ)."""
  cmd: ReachCommand = env.command_manager.get_term(command_name)  # type: ignore[assignment]
  quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).expand(env.num_envs, -1)
  return torch.cat([cmd.target_pos_b, quat], dim=-1)


def nv_last_action(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
  term: NvidiaDeltaAction = env.action_manager.get_term(action_name)  # type: ignore[assignment]
  return term.clipped_action
