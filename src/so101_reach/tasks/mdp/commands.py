"""Goal-conditioned reach command: target sampling, settle-based success, curriculum state.

The command owns everything that changes with the curriculum stage so the observation,
action, event and reward terms read one source of truth:

  * where targets come from (local neighbourhood of the TCP vs. whole workspace),
  * how the arm is initialised on reset,
  * success tolerance / hold time,
  * robustness settings (action noise, observation noise, action delay, gain DR),
  * smoothness-penalty scale.

Success ("settled") requires ``|tcp - target| < tol`` **and** ``|v_tcp| < max_settle_speed``
for ``hold_time_s`` consecutive seconds, so passing through the target fast never counts.
On success the command latches ``just_succeeded`` for one step (read by the success-bonus
reward) and, during training, immediately samples the next target from the current pose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv

from so101_reach.sim import so101
from so101_reach.sim.workspace import load_bank

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


@dataclass
class ReachStageCfg:
  name: str
  target_mode: Literal["local", "global"] = "global"
  local_radius: float = 0.08
  """Radius [m] around the current TCP for local targets."""
  p_local: float = 0.0
  """Probability of a local target in global mode (small corrective moves)."""
  init_random_prob: float = 0.0
  """Probability that a reset starts from a random feasible configuration."""
  init_rest_prob: float = 0.0
  """Probability that a reset starts from the folded rest pose."""
  success_tol: float = 0.02
  hold_time_s: float = 0.2
  action_noise_std: float = 0.0
  """Std of Gaussian noise added to the clipped action (action units)."""
  obs_noise_scale: float = 0.0
  """Multiplier on the per-term observation noise levels."""
  max_action_delay: int = 0
  """Maximum action delay in policy steps (20 ms each); sampled per episode."""
  kp_scale_range: tuple[float, float] = (1.0, 1.0)
  kv_scale_range: tuple[float, float] = (1.0, 1.0)
  smoothness_scale: float = 1.0
  promote_success_rate: float = 0.9
  """Settled-success rate (of the *stochastic* training policy) that advances the stage."""
  min_steps: int = 200 * 24
  """Minimum env steps (per env) spent in the stage before promotion."""
  max_steps: int | None = None
  """Promote unconditionally after this many env steps (exploration noise keeps the
  training success rate well below the deterministic policy's, so a pure threshold
  can stall the curriculum)."""


def default_stages() -> tuple[ReachStageCfg, ...]:
  return (
    ReachStageCfg(
      name="1_local_from_home", target_mode="local", local_radius=0.08, success_tol=0.02, max_steps=400 * 24
    ),
    ReachStageCfg(name="2_full_workspace", p_local=0.2, success_tol=0.015, promote_success_rate=0.75, max_steps=1200 * 24),
    ReachStageCfg(
      name="3_random_init",
      p_local=0.2,
      init_random_prob=0.7,
      init_rest_prob=0.1,
      success_tol=0.01,
      promote_success_rate=0.7,
      max_steps=1000 * 24,
    ),
    ReachStageCfg(
      name="4_robustness",
      p_local=0.2,
      init_random_prob=0.7,
      init_rest_prob=0.1,
      success_tol=0.01,
      action_noise_std=0.02,
      obs_noise_scale=1.0,
      max_action_delay=2,
      kp_scale_range=(0.7, 1.3),
      kv_scale_range=(0.7, 1.3),
      promote_success_rate=0.65,
      max_steps=1200 * 24,
    ),
    ReachStageCfg(
      name="5_precision",
      p_local=0.25,
      init_random_prob=0.7,
      init_rest_prob=0.1,
      success_tol=0.005,
      hold_time_s=0.3,
      action_noise_std=0.02,
      obs_noise_scale=1.0,
      max_action_delay=2,
      kp_scale_range=(0.7, 1.3),
      kv_scale_range=(0.7, 1.3),
      smoothness_scale=2.0,
      promote_success_rate=1.1,  # Final stage.
    ),
  )


class ReachCommand(CommandTerm):
  cfg: ReachCommandCfg

  def __init__(self, cfg: ReachCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    site_ids, _ = self.robot.find_sites((cfg.tcp_site,))
    self.tcp_site_id = site_ids[0]
    self.stages = tuple(cfg.stages)
    self.stage_idx = cfg.initial_stage
    self.stage_start_step = 0

    bank = load_bank(cfg.bank_split)
    self.bank_tcp = torch.tensor(bank.tcp_pos, device=self.device)
    self.bank_q = torch.tensor(bank.joint_pos, device=self.device)
    # Random subset used for local (nearest-neighbourhood) target queries.
    g = torch.Generator(device="cpu").manual_seed(0)
    sub = torch.randperm(len(bank), generator=g)[: cfg.local_query_size]
    self.bank_sub = self.bank_tcp[sub.to(self.device)]

    n = self.num_envs
    self.target_pos_b = torch.zeros(n, 3, device=self.device)
    self.hold_steps = torch.zeros(n, dtype=torch.long, device=self.device)
    self.just_succeeded = torch.zeros(n, dtype=torch.bool, device=self.device)
    self.target_steps = torch.zeros(n, dtype=torch.long, device=self.device)
    self.success_time_s = torch.zeros(n, device=self.device)
    self.ever_succeeded = torch.zeros(n, dtype=torch.bool, device=self.device)
    self.manual_target = torch.zeros(n, dtype=torch.bool, device=self.device)
    # Site kinematics are stale inside the reset path (forward() runs afterwards), so the
    # reset event stores the TCP of the configuration it wrote here.
    self.reset_tcp_hint = torch.zeros(n, 3, device=self.device)
    self._in_reset = False

    # Curriculum statistics (EMA over attempts that ended in success or timeout).
    self.ema_success = torch.zeros((), device=self.device)
    self.ema_attempts = torch.zeros((), device=self.device)

    for k in ("position_error", "tcp_speed", "settled", "success_rate", "time_to_success"):
      self.metrics[k] = torch.zeros(n, device=self.device)

  # Stage access.

  @property
  def stage(self) -> ReachStageCfg:
    return self.stages[self.stage_idx]

  def set_stage(self, idx: int) -> None:
    self.stage_idx = int(np.clip(idx, 0, len(self.stages) - 1))
    self.stage_start_step = self._env.common_step_counter
    self.ema_success.zero_()
    self.ema_attempts.zero_()

  @property
  def success_rate(self) -> torch.Tensor:
    return self.ema_success / self.ema_attempts.clamp(min=1e-6)

  # Geometry helpers (all in the robot base frame).

  @property
  def command(self) -> torch.Tensor:
    return self.target_pos_b

  @property
  def tcp_pos_b(self) -> torch.Tensor:
    p_w = self.robot.data.site_pos_w[:, self.tcp_site_id]
    root_p, root_q = self.robot.data.root_link_pos_w, self.robot.data.root_link_quat_w
    return quat_apply(quat_inv(root_q), p_w - root_p)

  @property
  def tcp_lin_vel_b(self) -> torch.Tensor:
    v_w = self.robot.data.site_lin_vel_w[:, self.tcp_site_id]
    return quat_apply(quat_inv(self.robot.data.root_link_quat_w), v_w)

  @property
  def target_pos_w(self) -> torch.Tensor:
    root_p, root_q = self.robot.data.root_link_pos_w, self.robot.data.root_link_quat_w
    return root_p + quat_apply(root_q, self.target_pos_b)

  @property
  def error_b(self) -> torch.Tensor:
    return self.target_pos_b - self.tcp_pos_b

  @property
  def distance(self) -> torch.Tensor:
    return torch.norm(self.error_b, dim=-1)

  @property
  def hold_steps_required(self) -> int:
    return max(1, int(round(self.stage.hold_time_s / self._env.step_dt)))

  # Target sampling.

  def _sample_global(self, n: int) -> torch.Tensor:
    idx = torch.randint(0, len(self.bank_tcp), (n,), device=self.device)
    return self.bank_tcp[idx]

  def _sample_local(self, centers: torch.Tensor, radius: float) -> torch.Tensor:
    d = torch.cdist(centers, self.bank_sub)  # (n, M)
    mask = d < radius
    score = torch.rand_like(d) * mask
    idx = score.argmax(dim=1)
    out = self.bank_sub[idx]
    has = mask.any(dim=1)
    if not bool(has.all()):
      out[~has] = self._sample_global(int((~has).sum()))
    return out

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    st = self.stage
    tcp = self.reset_tcp_hint[env_ids] if self._in_reset else self.tcp_pos_b[env_ids]
    if st.target_mode == "local":
      tgt = self._sample_local(tcp, st.local_radius)
    else:
      tgt = self._sample_global(n)
      if st.p_local > 0:
        loc = torch.rand(n, device=self.device) < st.p_local
        if bool(loc.any()):
          tgt[loc] = self._sample_local(tcp[loc], st.local_radius)
    keep = self.manual_target[env_ids]
    self.target_pos_b[env_ids] = torch.where(keep[:, None], self.target_pos_b[env_ids], tgt)
    self.hold_steps[env_ids] = 0
    self.target_steps[env_ids] = 0
    self.success_time_s[env_ids] = 0.0

  def set_target(self, target_pos_b: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
    """Command an explicit target (base frame). Used by evaluation and the interactive app."""
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device)
    self.target_pos_b[env_ids] = target_pos_b.to(self.device, torch.float32)
    self.manual_target[env_ids] = True
    self.hold_steps[env_ids] = 0
    self.target_steps[env_ids] = 0
    self.success_time_s[env_ids] = 0.0
    self.ever_succeeded[env_ids] = False
    self.time_left[env_ids] = float("inf")

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    assert isinstance(env_ids, torch.Tensor)
    self.ever_succeeded[env_ids] = False
    self.just_succeeded[env_ids] = False
    self._in_reset = True
    try:
      return super().reset(env_ids)
    finally:
      self._in_reset = False

  # Per-step update.

  def _update_metrics(self) -> None:
    dist = self.distance
    speed = torch.norm(self.tcp_lin_vel_b, dim=-1)
    st = self.stage
    inside = (dist < st.success_tol) & (speed < self.cfg.max_settle_speed)
    self.hold_steps = torch.where(inside, self.hold_steps + 1, torch.zeros_like(self.hold_steps))
    self.target_steps += 1
    success = self.hold_steps >= self.hold_steps_required
    self.just_succeeded = success & ~self.ever_succeeded
    self.success_time_s = torch.where(
      self.just_succeeded,
      (self.target_steps - self.hold_steps_required).float() * self._env.step_dt,
      self.success_time_s,
    )
    self.ever_succeeded |= success

    self.metrics["position_error"] = dist
    self.metrics["tcp_speed"] = speed
    self.metrics["settled"] = success.float()
    self.metrics["time_to_success"] = self.success_time_s

    if not self.cfg.resample_on_success:
      return
    # Attempts end on success or per-target timeout (time_left <= 0 this step).
    timeout = (self.time_left - self._env.step_dt <= 0.0) & ~self.just_succeeded
    a = self.cfg.ema_alpha
    self.ema_success += a * (self.just_succeeded.sum() - self.ema_success)
    self.ema_attempts += a * ((self.just_succeeded | timeout).sum() - self.ema_attempts)
    self.metrics["success_rate"][:] = self.success_rate
    done = self.just_succeeded.nonzero().flatten()
    if len(done) > 0:
      self.ever_succeeded[done] = False
      self._resample(done)

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    del env_ids

  # Visualisation.

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    for batch in visualizer.get_env_indices(self.num_envs):
      tgt = self.target_pos_w[batch].cpu().numpy()
      err = float(self.distance[batch])
      ok = err < self.stage.success_tol
      color = self.cfg.viz.success_color if ok else self.cfg.viz.target_color
      visualizer.add_sphere(center=tgt, radius=self.cfg.viz.target_radius, color=color, label=f"target_{batch}")
      tcp = self.robot.data.site_pos_w[batch, self.tcp_site_id].cpu().numpy()
      visualizer.add_sphere(center=tcp, radius=0.006, color=self.cfg.viz.tcp_color, label=f"tcp_{batch}")


@dataclass(kw_only=True)
class ReachCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  tcp_site: str = so101.TCP_SITE
  bank_split: str = "train"
  stages: tuple[ReachStageCfg, ...] = field(default_factory=default_stages)
  initial_stage: int = 0
  max_settle_speed: float = 0.05
  """TCP speed [m/s] below which the arm counts as settled."""
  resample_on_success: bool = True
  local_query_size: int = 100_000
  ema_alpha: float = 0.01

  @dataclass
  class VizCfg:
    target_color: tuple[float, float, float, float] = (1.0, 0.3, 0.1, 0.6)
    success_color: tuple[float, float, float, float] = (0.1, 0.9, 0.2, 0.6)
    tcp_color: tuple[float, float, float, float] = (0.1, 0.4, 1.0, 0.9)
    target_radius: float = 0.015

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> ReachCommand:
    return ReachCommand(self, env)
