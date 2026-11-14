"""Evaluation / interactive environment configs (identical physics for every controller).

* ``native``  : the trained policy's interface (25-D obs, incremental joint targets).
              Also used by the DLS-IK baseline, which acts through the same action term.
* ``nvidia``  : nvidia/so101-reach interface (20-D obs, measured-relative delta actions,
              gripper held at 0 per its contract).

Differences from training: no terminations (contacts and limit excursions are *measured*,
not reset), no rewards/curriculum, no observation/action noise, no delay, nominal servo
gains, targets set explicitly through ``ReachCommand.set_target``.
"""

from __future__ import annotations

from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from so101_reach.checkpoints.nvidia import load_contract
from so101_reach.tasks import mdp
from so101_reach.tasks.mdp import nvidia as nv
from so101_reach.tasks.reach_env_cfg import ACT, CMD, PHYSICS_DT, make_reach_env_cfg

NVIDIA_FIXED = {"wrist_roll": 0.0, "gripper": 0.0}


def make_eval_env_cfg(
  controller: Literal["native", "nvidia"] = "native",
  num_envs: int = 1,
  nvidia_rebase: Literal["policy_step", "every_10ms"] = "policy_step",
  nvidia_actuator: Literal["menagerie", "isaac_pd"] = "menagerie",
  actuator_model: Literal["menagerie", "identified"] = "menagerie",
) -> ManagerBasedRlEnvCfg:
  cfg = make_reach_env_cfg(play=True, actuator_model=actuator_model)
  cfg.scene.num_envs = num_envs
  cfg.rewards = {}
  cfg.curriculum = {}
  cfg.terminations = {"time_out": TerminationTermCfg(func=mdp.time_out, time_out=True)}
  cmd = cfg.commands[CMD]
  assert isinstance(cmd, mdp.ReachCommandCfg)
  cmd.resampling_time_range = (1e9, 1e9)
  cmd.bank_split = "eval"
  cfg.observations["critic"] = cfg.observations["actor"]

  if controller == "native":
    return cfg

  c = load_contract()
  cfg.actions = {
    ACT: nv.NvidiaDeltaActionCfg(
      entity_name="robot",
      scale=c.action_scale,
      max_step=c.max_step,
      joint_limits=c.joint_limits,
      fixed_targets=dict(NVIDIA_FIXED),
      rebase=nvidia_rebase,
      physics_dt=PHYSICS_DT,
    )
  }
  terms = {
    "joint_pos_rel": ObservationTermCfg(func=nv.nv_joint_pos_rel, params={"offsets": c.obs_joint_offsets}),
    "ee_position": ObservationTermCfg(func=nv.nv_ee_position, params={"command_name": CMD}),
    "generated_commands": ObservationTermCfg(func=nv.nv_goal_pose, params={"command_name": CMD}),
    "last_action": ObservationTermCfg(func=nv.nv_last_action, params={"action_name": ACT}),
  }
  group = ObservationGroupCfg(terms, enable_corruption=False)
  cfg.observations = {"actor": group, "critic": group}
  home = {n: c.default_joint_pos[n] for n in nv.NV_ACT_JOINTS}
  cfg.events = {
    "reset_arm": EventTermCfg(
      func=mdp.reset_arm_by_stage,
      mode="reset",
      params={"command_name": CMD, "home_noise": 0.0, "home_joint_pos": home, "fixed_joint_pos": NVIDIA_FIXED},
    ),
  }
  if nvidia_actuator == "isaac_pd":
    # Isaac Lab implicit-PD gains from the I/O descriptor (force limit stays at the
    # Menagerie STS3215 value; armature/friction stay Menagerie's).
    cfg.events["isaac_gains"] = EventTermCfg(
      func=mdp.set_servo_gains,
      mode="reset",
      params={"kp": dict(c.stiffness), "kv": dict(c.damping)},
    )
  return cfg
