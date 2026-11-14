"""SO-101 robot definition for MjLab, built on the MuJoCo Menagerie model.

The MJCF is vendored unmodified from mujoco_menagerie/robotstudio_so101 (commit in
``assets/robotstudio_so101/MENAGERIE_COMMIT``). Project-specific additions are applied
through ``MjSpec`` edits so the source file stays identical to upstream:

  * ``tcp`` site: the tool-center point. It is TheRobotStudio URDF ``gripper_frame_link``
    (``gripper_link`` offset [-0.0079, -0.000218, -0.0981] m), i.e. the tip of the closed
    jaws, and is the exact fingertip used by ``nvidia/so101-reach``. Menagerie's own
    ``gripperframe`` site lies ~20 mm further along the gripper x-axis (between open
    jaws) and is not used as the TCP. See ``scripts/verify_model.py``.
  * ``base_collision`` box: Menagerie gives the base body no collision geometry, so an
    arm folded over the base would pass through it. A conservative box (below the
    shoulder body) makes such configurations detectable.

Frames: world == robot base frame. The ``base`` body sits at the world origin with
identity orientation: +x forward (shoulder_pan = 0 points the arm along +x), +z up,
mounting plane at z = 0. Every Cartesian coordinate in this project is in this frame.
"""

from __future__ import annotations

from pathlib import Path

from typing import TYPE_CHECKING

import mujoco

if TYPE_CHECKING:
  from mjlab.entity import EntityCfg

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
from so101_reach.settings import SIM_CONFIG_DIR
ASSET_DIR: Path = PROJECT_ROOT / "assets" / "robotstudio_so101"
SO101_XML: Path = ASSET_DIR / "so101.xml"
SCENE_XML: Path = ASSET_DIR / "scene.xml"

ARM_JOINTS: tuple[str, ...] = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")
FIXED_JOINTS: tuple[str, ...] = ("wrist_roll", "gripper")
ALL_JOINTS: tuple[str, ...] = ARM_JOINTS + FIXED_JOINTS

TCP_SITE = "tcp"
NVIDIA_TIP_SITE = TCP_SITE  # Same point; kept as a separate name for readability.
NVIDIA_FINGERTIP_OFFSET: tuple[float, float, float] = (-0.0079, -0.000218, -0.0981)

# Ready pose: gripper pointing down, TCP at ~(0.236, 0.0, 0.132) m, all joints inside
# the 90 % soft limits.
HOME_JOINT_POS: dict[str, float] = {
  "shoulder_pan": 0.0,
  "shoulder_lift": -0.6,
  "elbow_flex": 0.3,
  "wrist_flex": 1.3,
  "wrist_roll": 0.0,
  "gripper": -0.15,
}

# Folded transport pose of the physical arm (TCP close to the table).
REST_JOINT_POS: dict[str, float] = {
  "shoulder_pan": 0.0,
  "shoulder_lift": -1.7,
  "elbow_flex": 1.6,
  "wrist_flex": 1.0,
  "wrist_roll": 0.0,
  "gripper": -0.15,
}

# Wrist roll and gripper are held during reaching. -0.15 rad closes the jaws (tip
# centres ~5 mm apart), so the TCP is the tip of the closed gripper.
FIXED_JOINT_TARGETS: dict[str, float] = {"wrist_roll": 0.0, "gripper": -0.15}

SOFT_JOINT_LIMIT_FACTOR = 0.9

BASE_COLLISION_POS = (0.014, 0.0, 0.03)
BASE_COLLISION_HALF_SIZE = (0.07, 0.056, 0.03)


def _add_project_elements(spec: mujoco.MjSpec) -> None:
  gripper = spec.body("gripper")
  gripper.add_site(
    name=TCP_SITE,
    pos=NVIDIA_FINGERTIP_OFFSET,
    size=(0.004, 0.004, 0.004),
    rgba=(0.1, 0.4, 1.0, 1.0),
    group=2,
  )
  base = spec.body("base")
  base.add_geom(
    name="base_collision",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=BASE_COLLISION_POS,
    size=BASE_COLLISION_HALF_SIZE,
    group=3,
    contype=1,
    conaffinity=1,
    rgba=(1.0, 0.0, 0.0, 0.3),
  )
  # MuJoCo's parent/child contact filter does not apply to children of world-welded
  # bodies; the pan servo legitimately sits inside the base.
  spec.add_exclude(bodyname1="base", bodyname2="shoulder")


SERVO_FIT_JSON: Path = PROJECT_ROOT / "configs" / "servo_model_fit.json"
# Joints without their own fit use the fit of this joint.
_FIT_FALLBACK = {"wrist_roll": "wrist_flex", "gripper": "wrist_flex"}


def servo_fit(path: Path | None = None) -> dict[str, dict[str, float]]:
  """Per-joint servo model identified on the real arm (scripts/fit_servo_model.py)."""
  import json

  joints = json.loads((path or SERVO_FIT_JSON).read_text())["joints"]
  return {n: joints[_FIT_FALLBACK.get(n, n)] for n in ALL_JOINTS}


def _apply_identified_servos(spec: mujoco.MjSpec, fit_path: Path | None = None) -> None:
  """Replace Menagerie's STS3215 gains by the model identified on the real arm (P = 16):
  position gain kp, velocity gain kv and joint friction loss per joint."""
  for n, f in servo_fit(fit_path).items():
    act = spec.actuator(n)
    act.gainprm[0] = f["kp"]
    act.biasprm[1] = -f["kp"]
    act.biasprm[2] = -f["kv"]
    spec.joint(n).frictionloss = f["frictionloss"]


ACTUATOR_MODELS = ("menagerie", "identified")


def get_spec(actuator_model: str = "menagerie") -> mujoco.MjSpec:
  """Robot-only spec (no floor); MjLab adds the terrain plane."""
  spec = mujoco.MjSpec.from_file(str(SO101_XML))
  _add_project_elements(spec)
  if actuator_model == "identified":
    _apply_identified_servos(spec)
  return spec


def make_cpu_model(actuator_model: str = "menagerie") -> mujoco.MjModel:
  """Menagerie scene (robot + floor) with project edits, for CPU MuJoCo tools."""
  spec = mujoco.MjSpec.from_file(str(SCENE_XML))
  _add_project_elements(spec)
  if actuator_model == "identified":
    _apply_identified_servos(spec)
  return spec.compile()


def get_robot_cfg(
  init_joint_pos: dict[str, float] | None = None,
  delay_max_lag: int = 0,
  actuator_model: str = "menagerie",
) -> EntityCfg:
  """MjLab entity with the XML position servos (XmlActuatorCfg).

  Args:
    init_joint_pos: Default joint configuration (defaults to HOME_JOINT_POS).
    delay_max_lag: Optional actuator command delay in physics steps (0 = none).
    actuator_model: "menagerie" (Menagerie STS3215 gains) or "identified" (fitted on the
      real arm, configs/servo_model_fit.json).
  """
  import functools

  from mjlab.actuator import XmlActuatorCfg
  from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.0),
      joint_pos=dict(init_joint_pos or HOME_JOINT_POS),
      joint_vel={".*": 0.0},
    ),
    spec_fn=functools.partial(get_spec, actuator_model),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        XmlActuatorCfg(
          target_names_expr=ALL_JOINTS,
          delay_min_lag=0,
          delay_max_lag=delay_max_lag,
        ),
      ),
      soft_joint_pos_limit_factor=SOFT_JOINT_LIMIT_FACTOR,
    ),
  )
