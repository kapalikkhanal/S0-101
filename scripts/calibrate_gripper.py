"""Calibrate the gripper's ticks -> simulator jaw angle from measured jaw openings (MOVES THE GRIPPER).

The arm holds its pose; only the gripper moves, one position at a time, when you press Enter.
At each position measure the gap between the two jaw TIPS (inner faces, at the very end of
the jaws) with a ruler or calipers and type it in mm. The zero and scale of the gripper's
joint map are then fitted so the Menagerie jaw geometry gives the same openings, and saved
to configs/real_joint_map.json (old file kept as .bak).

  python scripts/calibrate_gripper.py               # measure + fit + save (asks first)
  python scripts/calibrate_gripper.py --fit logs/real/gripper_calib_<time>.json   # refit only

Keep fingers out of the jaws while the gripper moves.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from so101_reach.real.joint_map import DEFAULT_MAP_PATH, RAD_PER_TICK, JointMap
from so101_reach.sim import so101
from so101_reach.sim.grasp_scene import SceneCfg, build_model

TARGETS = (2000, 2150, 2300, 2450, 2600, 2800)
LOG_DIR = so101.PROJECT_ROOT / "logs" / "real"


def sim_tip_gap() -> tuple[np.ndarray, np.ndarray]:
  """(angles, tip-to-tip gap [m]) of the Menagerie jaws over the gripper's joint range."""
  m = build_model(SceneCfg(objects=[]))
  d = mujoco.MjData(m)
  j = m.joint("gripper")
  ft = [m.geom(f"fixed_jaw_sph_tip{i}").id for i in (1, 2, 3)]
  mt = [m.geom(f"moving_jaw_sph_tip{i}").id for i in (1, 2, 3)]
  r = m.geom_size[ft[0]][0] + m.geom_size[mt[0]][0]
  angles = np.linspace(*m.jnt_range[j.id], 400)
  gaps = []
  for a in angles:
    d.qpos[m.jnt_qposadr[j.id]] = a
    mujoco.mj_kinematics(m, d)
    gaps.append(np.mean([np.linalg.norm(d.geom_xpos[f] - d.geom_xpos[g]) for f, g in zip(ft, mt, strict=True)]) - r)
  return angles, np.array(gaps)


def fit(meas: list[tuple[int, float]]) -> tuple[float, float, float]:
  """(zero_tick, scale, rms_mm) so that sim_gap(scale * (ticks - zero) * RAD_PER_TICK) = measured."""
  angles, gaps = sim_tip_gap()
  ticks = np.array([m[0] for m in meas], float)
  gap = np.array([m[1] for m in meas]) / 1000

  def res(x):
    a = x[1] * (ticks - x[0]) * RAD_PER_TICK
    return np.interp(a, angles, gaps) - gap

  sol = least_squares(res, [2048.0, 1.0], bounds=([1500, 0.5], [2600, 2.0]))
  return float(sol.x[0]), float(sol.x[1]), float(np.sqrt(np.mean(res(sol.x) ** 2)) * 1000)


def measure() -> Path:
  from so101_reach.real.feetech_bus import FeetechBus, load_calibration
  from so101_reach.real.joint_map import FOLLOWER_PORT
  from so101_reach.real.robot import RealPlant

  fcal = load_calibration()
  g = fcal["gripper"]
  meas = []
  with FeetechBus(FOLLOWER_PORT, calib=fcal) as bus:
    plant = RealPlant(bus, JointMap.load(), fcal, torque_limit=400)
    plant.engage()  # Arm holds its present pose.
    bus.write("gripper", "Torque_Limit", 300)
    print("The arm holds its pose; only the gripper moves. Keep fingers out of the jaws.")
    print("At each stop, measure the gap between the jaw TIPS (inner faces) in mm.\n")
    for target in TARGETS:
      target = int(np.clip(target, g.range_min + 20, g.range_max - 20))
      input(f"Enter = move the gripper to {target} ticks ")
      start = bus.sync_read("Present_Position")["gripper"]
      for k in range(1, 26):  # 0.5 s ramp
        bus.sync_write("Goal_Position", {"gripper": int(start + (target - start) * k / 25)})
        time.sleep(0.02)
      time.sleep(0.8)
      present = bus.sync_read("Present_Position")["gripper"]
      txt = input(f"  at {present} ticks - tip gap in mm (Enter = skip, q = done): ").strip()
      if txt.lower() == "q":
        break
      if txt:
        meas.append((int(present), float(txt)))
    plant.hold()
  LOG_DIR.mkdir(parents=True, exist_ok=True)
  path = LOG_DIR / f"gripper_calib_{datetime.now():%Y%m%d_%H%M%S}.json"
  path.write_text(json.dumps({"measurements": meas, "unit": "ticks, mm tip gap"}, indent=2) + "\n")
  print(f"saved measurements: {path}")
  return path


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--fit", type=Path, default=None, help="refit from a saved measurement file (no motion)")
  p.add_argument("--yes", action="store_true", help="save the fit without asking")
  args = p.parse_args()
  path = args.fit or measure()
  meas = [tuple(m) for m in json.loads(path.read_text())["measurements"]]
  if len(meas) < 3:
    raise SystemExit(f"need at least 3 measurements, got {len(meas)}")
  jmap = JointMap.load()
  old = jmap.joints["gripper"]
  zero, scale, rms = fit(meas)
  angles, gaps = sim_tip_gap()
  print(f"\nfit: zero_tick {zero:.0f} (was {old.zero_tick:.0f}), scale {scale:.3f} (was {old.scale:.3f}); residual {rms:.1f} mm RMS")
  print(f"{'ticks':>6} {'measured':>9} {'sim old':>8} {'sim new':>8}")
  for t, gmm in meas:
    a_old = old.sign * old.scale * (t - old.zero_tick) * RAD_PER_TICK
    a_new = scale * (t - zero) * RAD_PER_TICK
    print(f"{t:6d} {gmm:8.1f}mm {np.interp(a_old, angles, gaps) * 1e3:7.1f}mm {np.interp(a_new, angles, gaps) * 1e3:7.1f}mm")
  if args.yes or input("\nsave to configs/real_joint_map.json? [y/N] ").strip().lower() == "y":
    shutil.copy(DEFAULT_MAP_PATH, DEFAULT_MAP_PATH.with_suffix(".json.bak"))
    old.zero_tick, old.scale, old.sign = round(zero, 1), round(scale, 4), 1
    jmap.save()
    print(f"saved (previous: {DEFAULT_MAP_PATH.with_suffix('.json.bak')})")


if __name__ == "__main__":
  main()
