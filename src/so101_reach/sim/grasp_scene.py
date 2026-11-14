"""CPU MuJoCo scene for grasping: the SO-101 (identified servos) on a table with one free object.

Matches the real setup: the table plane at the measured height in the robot base frame
(about -13 mm), the servos' force capped by the Torque_Limit used on the real arm (STS3215
12 V, 30 kg.cm = 2.94 N.m at 1000/1000), the gripper servo with the identified STS3215 model
(same servo as the arm joints).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from so101_reach.sim import so101

STALL_TORQUE = 2.94  # N.m, STS3215 12 V (Torque_Limit 1000)
REAL_TORQUE_LIMITS = {"shoulder_pan": 400, "shoulder_lift": 400, "elbow_flex": 400, "wrist_flex": 400,
                      "wrist_roll": 400, "gripper": 350}


@dataclass
class ObjectCfg:
  shape: str = "sphere"  # sphere | box | cylinder
  size: tuple[float, ...] = (0.0258,)  # MuJoCo geom size (sphere: r; box: half sides; cylinder: r, half length)
  pos: tuple[float, float, float] = (0.2155, -0.1329, 0.0126)  # centre, robot base frame
  quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
  mass: float = 0.10
  friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
  rgba: tuple[float, float, float, float] = (1.0, 0.5, 0.0, 1.0)


@dataclass
class SceneCfg:
  table_z: float = -0.0132
  table_friction: float = 0.8
  torque_limits: dict[str, int] = field(default_factory=lambda: dict(REAL_TORQUE_LIMITS))
  objects: list[ObjectCfg] = field(default_factory=lambda: [ObjectCfg()])


_SHAPES = {"sphere": mujoco.mjtGeom.mjGEOM_SPHERE, "box": mujoco.mjtGeom.mjGEOM_BOX,
           "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER}


def build_spec(cfg: SceneCfg) -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(so101.SCENE_XML))
  so101._add_project_elements(spec)
  so101._apply_identified_servos(spec, so101.SIM_CONFIG_DIR / "servo_model_fit.json")
  for name, lim in cfg.torque_limits.items():
    act = spec.actuator(name)
    act.forcelimited = True
    act.forcerange = np.array([-1.0, 1.0]) * STALL_TORQUE * lim / 1000.0
  floor = spec.geom("floor")
  floor.pos = np.array([0.0, 0.0, cfg.table_z])
  floor.friction = np.array([cfg.table_friction, 0.005, 0.0001])
  for i, o in enumerate(cfg.objects):
    body = spec.worldbody.add_body(name=f"object{i}", pos=o.pos, quat=o.quat)
    body.add_freejoint(name=f"object{i}_joint")
    body.add_geom(name=f"object{i}_geom", type=_SHAPES[o.shape], size=list(o.size) + [0] * (3 - len(o.size)),
                  mass=o.mass, friction=o.friction, condim=4, rgba=o.rgba)
  return spec


def build_model(cfg: SceneCfg) -> mujoco.MjModel:
  return build_spec(cfg).compile()
