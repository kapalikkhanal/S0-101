"""Reward terms for settle-based reaching.

MjLab multiplies every term by ``weight * step_dt``. Dense terms are therefore "per
second"; one-off event terms (success bonus, collision) use ``weight = W / step_dt`` in the
config so that they pay exactly ``W`` per event.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

  from so101_reach.tasks.mdp.commands import ReachCommand


def _cmd(env: ManagerBasedRlEnv, command_name: str) -> ReachCommand:
  return env.command_manager.get_term(command_name)  # type: ignore[return-value]


def tcp_distance(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Euclidean TCP-target distance [m] (use a negative weight)."""
  return _cmd(env, command_name).distance


def tcp_distance_tanh(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
  """1 - tanh(d / std): bounded attraction kernel; small std = precision term."""
  return 1.0 - torch.tanh(_cmd(env, command_name).distance / std)


def tcp_distance_exp(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
  """exp(-d / std): sharp precision kernel (mm-scale std)."""
  return torch.exp(-_cmd(env, command_name).distance / std)


def near_target_speed(env: ManagerBasedRlEnv, command_name: str, radius: float) -> torch.Tensor:
  """|v_tcp|^2 inside `radius` of the target (anti fly-through / overshoot)."""
  cmd = _cmd(env, command_name)
  speed2 = torch.sum(torch.square(cmd.tcp_lin_vel_b), dim=-1)
  return speed2 * (cmd.distance < radius).float()


def success_bonus(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """1 on the step after a target was reached and held (settled)."""
  return _cmd(env, command_name).just_succeeded.float()


def terminated_non_timeout(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.termination_manager.terminated.float()
