"""Observation terms. All Cartesian quantities are in the robot base frame.

Noise is applied here (not through MjLab's corruption config) because its magnitude is
set by the curriculum stage: ``noise_std * stage.obs_noise_scale``. Play/eval configs pin a
stage without noise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from so101_reach.sim import so101

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

  from so101_reach.tasks.mdp.actions import IncrementalJointPositionAction
  from so101_reach.tasks.mdp.commands import ReachCommand

_ARM_CFG = SceneEntityCfg("robot", joint_names=so101.ARM_JOINTS, preserve_order=True)


def _cmd(env: ManagerBasedRlEnv, command_name: str) -> ReachCommand:
  return env.command_manager.get_term(command_name)  # type: ignore[return-value]


def _noisy(x: torch.Tensor, env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
  scale = _cmd(env, command_name).stage.obs_noise_scale * std
  return x + scale * torch.randn_like(x) if scale > 0 else x


def arm_joint_pos(
  env: ManagerBasedRlEnv, command_name: str, noise_std: float = 0.0, asset_cfg: SceneEntityCfg = _ARM_CFG
) -> torch.Tensor:
  """Absolute arm joint angles [rad] (default pose is not subtracted)."""
  robot: Entity = env.scene[asset_cfg.name]
  return _noisy(robot.data.joint_pos[:, asset_cfg.joint_ids], env, command_name, noise_std)


def arm_joint_vel(
  env: ManagerBasedRlEnv, command_name: str, noise_std: float = 0.0, asset_cfg: SceneEntityCfg = _ARM_CFG
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  return _noisy(robot.data.joint_vel[:, asset_cfg.joint_ids], env, command_name, noise_std)


def tcp_pos(env: ManagerBasedRlEnv, command_name: str, noise_std: float = 0.0) -> torch.Tensor:
  return _noisy(_cmd(env, command_name).tcp_pos_b, env, command_name, noise_std)


def target_pos(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return _cmd(env, command_name).target_pos_b


def target_error(env: ManagerBasedRlEnv, command_name: str, noise_std: float = 0.0) -> torch.Tensor:
  """target - tcp [m]."""
  return _noisy(_cmd(env, command_name).error_b, env, command_name, noise_std)


def tcp_lin_vel(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return _cmd(env, command_name).tcp_lin_vel_b


def clipped_last_action(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
  term: IncrementalJointPositionAction = env.action_manager.get_term(action_name)  # type: ignore[assignment]
  return term.clipped_action


def target_lead(env: ManagerBasedRlEnv, action_name: str, asset_cfg: SceneEntityCfg = _ARM_CFG) -> torch.Tensor:
  """Commanded joint target minus measured joint position [rad]."""
  term: IncrementalJointPositionAction = env.action_manager.get_term(action_name)  # type: ignore[assignment]
  robot: Entity = env.scene[asset_cfg.name]
  return term.joint_target - robot.data.joint_pos[:, asset_cfg.joint_ids]


# Privileged (critic-only) terms.


def hold_progress(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  cmd = _cmd(env, command_name)
  return (cmd.hold_steps.float() / cmd.hold_steps_required).clamp(max=1.0).unsqueeze(-1)


def success_tolerance(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  cmd = _cmd(env, command_name)
  return torch.full((env.num_envs, 1), cmd.stage.success_tol, device=env.device)


def action_delay(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
  term: IncrementalJointPositionAction = env.action_manager.get_term(action_name)  # type: ignore[assignment]
  return term.lag.float().unsqueeze(-1)


def actuator_kp(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Per-env position gain of the arm actuators relative to the nominal value."""
  robot: Entity = env.scene["robot"]
  ctrl_ids = torch.cat([a.global_ctrl_ids for a in robot.actuators])[:4]
  kp = env.sim.model.actuator_gainprm[:, ctrl_ids, 0]
  if kp.dim() == 1:
    kp = kp.unsqueeze(0).expand(env.num_envs, -1)
  return kp / env.sim.get_default_field("actuator_gainprm")[ctrl_ids, 0]
