"""SO-101 goal-conditioned reaching environment (MjLab manager-based RL env).

Control: 50 Hz policy (5 ms physics x decimation 4), bounded joint-target increments for
the four arm joints; wrist roll and gripper held (closed gripper).

Actor observation (25-D, base frame):
  arm joint pos (4) | arm joint vel (4) | TCP pos (3) | target pos (3) | target - TCP (3)
  | last clipped action (4) | commanded target - measured joint pos (4)
Critic: the same without noise + TCP velocity, hold progress, success tolerance,
action delay, servo-gain scale (privileged).
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from so101_reach.sim import so101
from so101_reach.tasks import mdp

PHYSICS_DT = 0.005
DECIMATION = 4
STEP_DT = PHYSICS_DT * DECIMATION  # 50 Hz policy.
CMD = "ee_target"
ACT = "arm"

_ARM = SceneEntityCfg("robot", joint_names=so101.ARM_JOINTS, preserve_order=True)


def _contact_sensors() -> tuple[ContactSensorCfg, ...]:
  ground = ContactSensorCfg(
    name="ground_contact",
    primary=ContactMatch(mode="subtree", pattern="shoulder", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    history_length=DECIMATION,
  )
  self_contact = ContactSensorCfg(
    name="self_contact",
    primary=ContactMatch(mode="body", pattern=("base", "shoulder", "upper_arm"), entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="lower_arm", entity="robot"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    history_length=DECIMATION,
  )
  return (ground, self_contact)


def _policy_terms(noisy: bool) -> dict[str, ObservationTermCfg]:
  s = 1.0 if noisy else 0.0
  return {
    "joint_pos": ObservationTermCfg(
      func=mdp.arm_joint_pos, params={"command_name": CMD, "noise_std": 0.003 * s, "asset_cfg": _ARM}
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.arm_joint_vel, params={"command_name": CMD, "noise_std": 0.05 * s, "asset_cfg": _ARM}
    ),
    "tcp_pos": ObservationTermCfg(func=mdp.tcp_pos, params={"command_name": CMD, "noise_std": 0.001 * s}),
    "target_pos": ObservationTermCfg(func=mdp.target_pos, params={"command_name": CMD}),
    "target_error": ObservationTermCfg(func=mdp.target_error, params={"command_name": CMD, "noise_std": 0.001 * s}),
    "last_action": ObservationTermCfg(func=mdp.clipped_last_action, params={"action_name": ACT}),
    "target_lead": ObservationTermCfg(func=mdp.target_lead, params={"action_name": ACT, "asset_cfg": _ARM}),
  }


def make_reach_env_cfg(
  play: bool = False, stage: int | None = None, actuator_model: str = "menagerie", servo_dr: bool = True
) -> ManagerBasedRlEnvCfg:
  """Training config (curriculum from stage 1) or play config (fixed stage, no noise/DR).

  Args:
    play: Play/eval mode: curriculum disabled, no resampling on success, infinite episode.
    stage: Fixed 0-based stage index; play defaults to the final stage's tolerance
      settings with robustness disabled.
    actuator_model: "menagerie" or "identified" (servo model fitted on the real arm). With
      "identified", the robustness stages randomise servo kp/kv x[0.6, 1.6] and joint
      friction x[0.5, 2.0] around the fit.
    servo_dr: False disables all servo randomization (kp/kv/friction stay at the model's
      values; delay, action and observation noise are kept). Ablation: identified model
      without randomization vs randomization without identification vs both.
  """
  stages = mdp.default_stages()
  for st in stages:
    if not servo_dr:
      st.kp_scale_range = st.kv_scale_range = (1.0, 1.0)
    elif actuator_model == "identified" and st.kp_scale_range != (1.0, 1.0):
      st.kp_scale_range = (0.6, 1.6)
      st.kv_scale_range = (0.6, 1.6)
  if play:
    # Deterministic evaluation conditions: final-stage tolerance, no noise / delay / DR.
    final = stages[-1]
    stages = (
      mdp.ReachStageCfg(
        name="play",
        p_local=final.p_local,
        success_tol=final.success_tol,
        hold_time_s=final.hold_time_s,
        promote_success_rate=1.1,
      ),
    )
    stage = 0

  commands: dict[str, CommandTermCfg] = {
    CMD: mdp.ReachCommandCfg(
      resampling_time_range=(5.0, 5.0),
      debug_vis=True,
      stages=stages,
      initial_stage=stage or 0,
      bank_split="train",
    )
  }

  critic_terms = {
    **_policy_terms(noisy=False),
    "tcp_vel": ObservationTermCfg(func=mdp.tcp_lin_vel, params={"command_name": CMD}),
    "hold_progress": ObservationTermCfg(func=mdp.hold_progress, params={"command_name": CMD}),
    "success_tol": ObservationTermCfg(func=mdp.success_tolerance, params={"command_name": CMD}),
    "action_delay": ObservationTermCfg(func=mdp.action_delay, params={"action_name": ACT}),
    "servo_kp": ObservationTermCfg(func=mdp.actuator_kp),
  }
  observations = {
    "actor": ObservationGroupCfg(_policy_terms(noisy=True), enable_corruption=False, nan_policy="sanitize"),
    "critic": ObservationGroupCfg(critic_terms, enable_corruption=False, nan_policy="sanitize"),
  }

  actions = {ACT: mdp.IncrementalJointPositionActionCfg(entity_name="robot", command_name=CMD)}

  events = {
    "reset_arm": EventTermCfg(func=mdp.reset_arm_by_stage, mode="reset", params={"command_name": CMD}),
    "servo_gains": EventTermCfg(func=mdp.randomize_servo_gains_by_stage, mode="reset", params={"command_name": CMD}),
  }
  if actuator_model == "identified" and servo_dr and not play:
    events["joint_friction"] = EventTermCfg(
      func=mdp.dr.joint_friction,
      mode="reset",
      params={"asset_cfg": _ARM, "ranges": (0.5, 2.0), "operation": "scale"},
    )

  rewards = {
    "tcp_distance": RewardTermCfg(func=mdp.tcp_distance, weight=-2.0, params={"command_name": CMD}),
    "reach_coarse": RewardTermCfg(func=mdp.tcp_distance_tanh, weight=1.0, params={"command_name": CMD, "std": 0.1}),
    "reach_fine": RewardTermCfg(func=mdp.tcp_distance_tanh, weight=1.0, params={"command_name": CMD, "std": 0.02}),
    "reach_precise": RewardTermCfg(func=mdp.tcp_distance_exp, weight=2.0, params={"command_name": CMD, "std": 0.005}),
    "success_bonus": RewardTermCfg(func=mdp.success_bonus, weight=2.0 / STEP_DT, params={"command_name": CMD}),
    "near_target_speed": RewardTermCfg(
      func=mdp.near_target_speed, weight=-2.0, params={"command_name": CMD, "radius": 0.03}
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.02),
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-5e-6, params={"asset_cfg": _ARM}),
    "joint_vel_l2": RewardTermCfg(func=mdp.joint_vel_l2, weight=-0.005, params={"asset_cfg": _ARM}),
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-5.0, params={"asset_cfg": _ARM}),
    "collision": RewardTermCfg(func=mdp.terminated_non_timeout, weight=-5.0 / STEP_DT),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "ground_contact": TerminationTermCfg(
      func=mdp.illegal_contact, params={"sensor_name": "ground_contact", "force_threshold": 1.0}
    ),
    "self_collision": TerminationTermCfg(
      func=mdp.illegal_contact, params={"sensor_name": "self_contact", "force_threshold": 1.0}
    ),
    "nan": TerminationTermCfg(func=mdp.nan_detection),
  }

  curriculum = {
    "stage": CurriculumTermCfg(
      func=mdp.reach_stage_curriculum,
      params={
        "command_name": CMD,
        "smoothness_terms": ("action_rate_l2", "joint_acc_l2"),
        "enabled": not play,
      },
    )
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": so101.get_robot_cfg(actuator_model=actuator_model)},
      sensors=_contact_sensors(),
      num_envs=1,
      env_spacing=1.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="base",
      distance=0.75,
      elevation=-25.0,
      azimuth=135.0,
      height=720,
      width=1280,
    ),
    sim=SimulationCfg(
      nconmax=40,
      njmax=300,
      mujoco=MujocoCfg(
        timestep=PHYSICS_DT,
        integrator="implicitfast",
        iterations=10,
        ls_iterations=20,
        impratio=10,
        cone="elliptic",
      ),
    ),
    decimation=DECIMATION,
    episode_length_s=12.0,
  )

  if play:
    cfg.episode_length_s = 1e9
    cmd = cfg.commands[CMD]
    assert isinstance(cmd, mdp.ReachCommandCfg)
    cmd.resampling_time_range = (6.0, 6.0)
    cmd.resample_on_success = False
  return cfg
