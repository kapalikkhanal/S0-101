"""Mapping between STS3215 raw ticks (LeRobot-calibrated follower) and simulator joint angles.

    q_sim [rad] = sign * scale * (ticks - zero_tick) * 2*pi / 4096

LeRobot's calibration writes a homing offset into each servo so that the arm's middle pose
reads ~2048 ticks; TheRobotStudio `so101_new_calib` URDF (and therefore the Menagerie MJCF
used in simulation) puts its joint zeros at that same middle pose. Defaults are therefore
zero_tick = 2048, sign = +1. Verify with `scripts/real_twin.py` (direction check + tap test)
and edit `configs/real_joint_map.json` if a joint is mirrored or offset. The gripper's
zero and scale are fitted to the jaw opening (scripts/calibrate_gripper.py): its LeRobot range
does not line up with the Menagerie jaw geometry.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from so101_reach.sim import so101

TICKS_PER_REV = 4096
RAD_PER_TICK = 2 * math.pi / TICKS_PER_REV
DEFAULT_MAP_PATH: Path = so101.PROJECT_ROOT / "configs" / "real_joint_map.json"
from so101_reach.settings import FOLLOWER_PORT


@dataclass
class JointMapEntry:
  sign: int = 1
  zero_tick: float = 2048.0
  scale: float = 1.0


@dataclass
class JointMap:
  joints: dict[str, JointMapEntry]

  @staticmethod
  def default() -> JointMap:
    return JointMap({n: JointMapEntry() for n in so101.ALL_JOINTS})

  @staticmethod
  def load(path: Path = DEFAULT_MAP_PATH) -> JointMap:
    if not path.exists():
      return JointMap.default()
    raw = json.loads(path.read_text())
    keys = ("sign", "zero_tick", "scale")
    return JointMap({n: JointMapEntry(**{k: v for k, v in raw[n].items() if k in keys}) for n in so101.ALL_JOINTS})

  def save(self, path: Path = DEFAULT_MAP_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({n: asdict(e) for n, e in self.joints.items()}, indent=2) + "\n")

  def to_rad(self, name: str, ticks: float) -> float:
    e = self.joints[name]
    return e.sign * e.scale * (ticks - e.zero_tick) * RAD_PER_TICK

  def to_ticks(self, name: str, rad: float) -> int:
    e = self.joints[name]
    return int(round(e.zero_tick + e.sign * rad / (e.scale * RAD_PER_TICK)))
