"""Performance-gated curriculum over the stages defined on the reach command.

A stage is promoted when (a) at least ``stage.min_steps`` env steps were spent in it and
(b) the EMA settled-success rate of target attempts reaches ``stage.promote_success_rate``,
or unconditionally once ``stage.max_steps`` env steps were spent in it.
On promotion the smoothness penalty weights are rescaled by ``stage.smoothness_scale``.
The stage index is persisted in checkpoints by ``So101ReachRunner``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

  from so101_reach.tasks.mdp.commands import ReachCommand


class reach_stage_curriculum:
  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self._base_weights = {
      name: env.reward_manager.get_term_cfg(name).weight for name in cfg.params["smoothness_terms"]
    }
    self._last_check = -1
    self._applied_stage = -1

  def _apply_weights(self, env: ManagerBasedRlEnv, cmd: ReachCommand) -> None:
    for name, w in self._base_weights.items():
      env.reward_manager.get_term_cfg(name).weight = w * cmd.stage.smoothness_scale
    self._applied_stage = cmd.stage_idx

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice,
    command_name: str,
    smoothness_terms: tuple[str, ...],
    enabled: bool = True,
    check_every: int = 24,
  ) -> dict[str, torch.Tensor]:
    del env_ids, smoothness_terms
    cmd: ReachCommand = env.command_manager.get_term(command_name)  # type: ignore[assignment]
    step = env.common_step_counter
    if enabled and step - self._last_check >= check_every:
      self._last_check = step
      st = cmd.stage
      dwell = step - cmd.stage_start_step
      performed = (
        dwell >= st.min_steps
        and float(cmd.success_rate) >= st.promote_success_rate
        and float(cmd.ema_attempts) > 1.0
      )
      timed_out = st.max_steps is not None and dwell >= st.max_steps
      if cmd.stage_idx < len(cmd.stages) - 1 and (performed or timed_out):
        why = "success" if performed else "max dwell"
        print(f"[curriculum] step {step}: stage {st.name} -> {cmd.stages[cmd.stage_idx + 1].name} ({why}, success {float(cmd.success_rate):.3f})")
        cmd.set_stage(cmd.stage_idx + 1)
    if self._applied_stage != cmd.stage_idx:
      self._apply_weights(env, cmd)
    return {
      "stage": torch.tensor(float(cmd.stage_idx + 1)),
      "success_rate": cmd.success_rate.detach().clone(),
      "success_tol_mm": torch.tensor(cmd.stage.success_tol * 1e3),
    }
