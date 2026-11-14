"""Bounded joint-target increments for the SO-101 position servos.

Each policy step (50 Hz):

  a_c      = clip(a, -1, 1) (+ N(0, sigma) during the robustness stages)
  delta    = scale * a_c                         (at most `scale` rad per step and joint)
  target  <- clip(target + delta_{t - lag}, soft joint limits)  (inward-only if outside)
  target  <- clip(target, q - max_lead, q + max_lead)

and the target is sent to the servos at every physics substep. Integrating a commanded
target (instead of adding the delta to the *measured* position) makes zero action mean
"hold", keeps the command smooth, and maps 1:1 onto the Goal_Position register of the
physical STS3215 servos. The ``max_lead`` clamp bounds wind-up when the arm is blocked.

The wrist roll and gripper are re-commanded to fixed targets every substep (MjLab zeroes
actuator targets on reset).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg
from mjlab.actuator.actuator import TransmissionType

from so101_reach.sim import so101

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

  from so101_reach.tasks.mdp.commands import ReachCommand


@dataclass(kw_only=True)
class IncrementalJointPositionActionCfg(BaseActionCfg):
  actuator_names: tuple[str, ...] = so101.ARM_JOINTS
  scale: float = 0.04
  """Maximum joint-target increment per policy step [rad] (2 rad/s at 50 Hz)."""
  max_lead: float = 0.15
  """Maximum distance between commanded target and measured joint position [rad]."""
  fixed_targets: dict[str, float] = field(default_factory=lambda: dict(so101.FIXED_JOINT_TARGETS))
  command_name: str | None = "ee_target"
  """Command term holding the curriculum stage (action noise, delay). None disables."""
  max_delay_buffer: int = 4
  preserve_order: bool = True

  def __post_init__(self):
    self.transmission_type = TransmissionType.JOINT

  def build(self, env: ManagerBasedRlEnv) -> IncrementalJointPositionAction:
    return IncrementalJointPositionAction(self, env)


class IncrementalJointPositionAction(BaseAction):
  cfg: IncrementalJointPositionActionCfg

  def __init__(self, cfg: IncrementalJointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    self._env = env
    n, d = self.num_envs, self.action_dim
    lim = self._entity.data.soft_joint_pos_limits[:, self._target_ids]
    self._lo, self._hi = lim[..., 0].clone(), lim[..., 1].clone()
    self._joint_target = torch.zeros(n, d, device=self.device)
    self._clipped = torch.zeros(n, d, device=self.device)
    self._delta_hist = torch.zeros(n, cfg.max_delay_buffer + 1, d, device=self.device)
    self._lag = torch.zeros(n, dtype=torch.long, device=self.device)
    fixed_ids, fixed_names = self._entity.find_joints(tuple(cfg.fixed_targets.keys()), preserve_order=True)
    self._fixed_ids = torch.tensor(fixed_ids, device=self.device, dtype=torch.long)
    self._fixed_val = torch.tensor([cfg.fixed_targets[k] for k in fixed_names], device=self.device).expand(n, -1)

  @property
  def joint_target(self) -> torch.Tensor:
    return self._joint_target

  @property
  def clipped_action(self) -> torch.Tensor:
    return self._clipped

  @property
  def lag(self) -> torch.Tensor:
    return self._lag

  def _command(self) -> ReachCommand | None:
    if self.cfg.command_name is None:
      return None
    return self._env.command_manager.get_term(self.cfg.command_name)  # type: ignore[return-value]

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions
    a = torch.nan_to_num(actions, nan=0.0).clamp(-1.0, 1.0)
    self._clipped[:] = a
    cmd = self._command()
    if cmd is not None and cmd.stage.action_noise_std > 0:
      a = (a + cmd.stage.action_noise_std * torch.randn_like(a)).clamp(-1.0, 1.0)
    delta = a * self.cfg.scale
    self._delta_hist = torch.roll(self._delta_hist, shifts=1, dims=1)
    self._delta_hist[:, 0] = delta
    delta = self._delta_hist[torch.arange(self.num_envs, device=self.device), self._lag]
    q = self._entity.data.joint_pos[:, self._target_ids]
    prev = self._joint_target
    # A target that starts outside the soft limits (e.g. the folded rest pose) may only
    # move inward; it is never snapped to the limit in one step.
    target = torch.clamp(prev + delta, torch.minimum(self._lo, prev), torch.maximum(self._hi, prev))
    target = torch.clamp(target, q - self.cfg.max_lead, q + self.cfg.max_lead)
    self._joint_target[:] = target
    self._processed_actions = target

  def apply_actions(self) -> None:
    self._entity.set_joint_position_target(self._joint_target, joint_ids=self._target_ids)
    self._entity.set_joint_position_target(self._fixed_val, joint_ids=self._fixed_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_actions[env_ids] = 0.0
    self._clipped[env_ids] = 0.0
    self._delta_hist[env_ids] = 0.0
    # Joint state written by reset events is already visible in data.joint_pos.
    self._joint_target[env_ids] = self._entity.data.joint_pos[env_ids][:, self._target_ids]
    cmd = self._command()
    max_lag = min(cmd.stage.max_action_delay, self.cfg.max_delay_buffer) if cmd is not None else 0
    n = self._lag[env_ids].shape[0]
    self._lag[env_ids] = torch.randint(0, max_lag + 1, (n,), device=self.device)

  def sync_target_to_measured(self, env_ids: torch.Tensor | None = None) -> None:
    """Set the commanded target to the measured joint positions (e.g. after teleporting)."""
    ids = slice(None) if env_ids is None else env_ids
    self._joint_target[ids] = self._entity.data.joint_pos[ids][:, self._target_ids]
