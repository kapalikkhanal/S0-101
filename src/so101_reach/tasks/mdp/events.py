"""Reset events: stage-dependent initial configuration and actuator-gain randomisation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

from so101_reach.sim import so101

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

  from so101_reach.tasks.mdp.commands import ReachCommand


def fk_tcp_cpu(joint_pos: dict[str, float]) -> np.ndarray:
  """TCP position (base frame) of a joint configuration, via CPU MuJoCo."""
  model = so101.make_cpu_model()
  data = mujoco.MjData(model)
  for n, v in joint_pos.items():
    data.qpos[model.jnt_qposadr[model.joint(n).id]] = v
  mujoco.mj_kinematics(model, data)
  return data.site_xpos[model.site(so101.TCP_SITE).id].copy()


class reset_arm_by_stage:
  """Reset the arm to home, the rest pose or a random feasible bank configuration.

  Probabilities come from the command's current stage. Bank configurations are
  collision-free with clearance (see so101_reach.sim.workspace). Writes the TCP of the
  chosen configuration to ``command.reset_tcp_hint`` for local target sampling.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self.robot: Entity = env.scene["robot"]
    arm_ids, _ = self.robot.find_joints(so101.ARM_JOINTS, preserve_order=True)
    fix_ids, _ = self.robot.find_joints(so101.FIXED_JOINTS, preserve_order=True)
    self.arm_ids = torch.tensor(arm_ids, device=env.device)
    self.fix_ids = torch.tensor(fix_ids, device=env.device)
    dev = env.device
    home = {**so101.HOME_JOINT_POS, **(cfg.params.get("home_joint_pos") or {})}
    fixed = {**so101.FIXED_JOINT_TARGETS, **(cfg.params.get("fixed_joint_pos") or {})}
    rest = {**so101.REST_JOINT_POS, **fixed}
    self.home_q = torch.tensor([home[n] for n in so101.ARM_JOINTS], device=dev)
    self.rest_q = torch.tensor([rest[n] for n in so101.ARM_JOINTS], device=dev)
    self.fixed_q = torch.tensor([fixed[n] for n in so101.FIXED_JOINTS], device=dev)
    self.home_tcp = torch.tensor(fk_tcp_cpu({**home, **fixed}), device=dev, dtype=torch.float32)
    self.rest_tcp = torch.tensor(fk_tcp_cpu(rest), device=dev, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    command_name: str,
    home_noise: float = 0.02,
    home_joint_pos: dict[str, float] | None = None,
    fixed_joint_pos: dict[str, float] | None = None,
  ) -> None:
    del home_joint_pos, fixed_joint_pos  # Consumed in __init__.
    if env_ids is None:
      env_ids = torch.arange(env.num_envs, device=env.device)
    cmd: ReachCommand = env.command_manager.get_term(command_name)  # type: ignore[assignment]
    st = cmd.stage
    n = len(env_ids)
    u = torch.rand(n, device=env.device)
    use_rand = u < st.init_random_prob
    use_rest = (u >= st.init_random_prob) & (u < st.init_random_prob + st.init_rest_prob)

    q = self.home_q.expand(n, -1) + home_noise * (2 * torch.rand(n, 4, device=env.device) - 1)
    tcp = self.home_tcp.expand(n, -1).clone()
    q = torch.where(use_rest[:, None], self.rest_q.expand(n, -1), q)
    tcp = torch.where(use_rest[:, None], self.rest_tcp.expand(n, -1), tcp)
    if bool(use_rand.any()):
      idx = torch.randint(0, len(cmd.bank_q), (int(use_rand.sum()),), device=env.device)
      q[use_rand] = cmd.bank_q[idx]
      tcp[use_rand] = cmd.bank_tcp[idx]

    full_q = torch.cat([q, self.fixed_q.expand(n, -1)], dim=1)
    ids = torch.cat([self.arm_ids, self.fix_ids])
    self.robot.write_joint_state_to_sim(full_q, torch.zeros_like(full_q), env_ids=env_ids, joint_ids=ids)
    cmd.reset_tcp_hint[env_ids] = tcp


@requires_model_fields("actuator_gainprm", "actuator_biasprm")
def set_servo_gains(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  kp: dict[str, float],
  kv: dict[str, float],
) -> None:
  """Set absolute position-servo gains per joint (MuJoCo <position>: gain=kp, bias=[0,-kp,-kv])."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  robot: Entity = env.scene["robot"]
  ctrl_ids = torch.cat([a.global_ctrl_ids for a in robot.actuators])
  names = [n for a in robot.actuators for n in a.target_names]
  model = env.sim.model
  for j, c in zip(names, ctrl_ids.tolist(), strict=True):
    if j in kp:
      model.actuator_gainprm[env_ids, c, 0] = kp[j]
      model.actuator_biasprm[env_ids, c, 1] = -kp[j]
      model.actuator_biasprm[env_ids, c, 2] = -kv[j]


@requires_model_fields("actuator_gainprm", "actuator_biasprm")
def randomize_servo_gains_by_stage(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  command_name: str,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", actuator_names=(".*",)),
) -> None:
  """Scale servo kp/kv by the stage's ranges (identity before the robustness stage)."""
  cmd: ReachCommand = env.command_manager.get_term(command_name)  # type: ignore[assignment]
  st = cmd.stage
  dr.pd_gains(
    env,
    env_ids,
    kp_range=st.kp_scale_range,
    kd_range=st.kv_scale_range,
    asset_cfg=asset_cfg,
    operation="scale",
  )
