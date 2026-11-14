"""The REAL SO-101 follows the mouse pointer (MOVES THE ARM).

  python scripts/real_mouse.py                    # 25 % speed, fine-tuned policy
  python scripts/real_mouse.py --speed 0.5        # fraction of the full speed limit (0.05-1.0)

The window shows a live twin of the arm. It starts FROZEN (holding): left click or F starts /
stops following. The cursor is projected on a horizontal plane (scroll = height, 8-45 cm);
targets are clipped to configs/real_workspace.json (wall keep-out drawn in red), the
minimum height and the verified workspace. +/- change the speed live (5 % steps). Esc holds
the pose and quits (torque stays on; scripts/real_release.py lets the arm go limp).

  python scripts/real_mouse.py --sim-bus --headless-test   # control-core test, no arm/window
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ["MUJOCO_GL"] = "glfw"

import numpy as np  # noqa: E402

from so101_reach.real.controller import ControllerCfg, ReachController  # noqa: E402
from so101_reach.real.feetech_bus import FeetechBus, load_calibration  # noqa: E402
from so101_reach.real.joint_map import FOLLOWER_PORT, JointMap  # noqa: E402
from so101_reach.real.mouse_real import MouseFollowCore, SafeTargets, run_window  # noqa: E402
from so101_reach.real.robot import RealPlant, SimRealPlant, SimServoBus  # noqa: E402
from so101_reach.sim import so101  # noqa: E402

from so101_reach.settings import POLICY_DIR as POLICY


def headless_test(core: MouseFollowCore) -> None:
  """Scripted cursor: a figure-eight sweeping into the wall keep-out and down to the table."""
  core.toggle()
  errs, notes = [], set()
  for k in range(int(12 / core.plant.dt)):
    t = k * core.plant.dt
    cursor = np.array([0.22 + 0.06 * np.sin(2 * np.pi * t / 6), 0.20 * np.sin(2 * np.pi * t / 3), 0.10 + 0.06 * np.cos(2 * np.pi * t / 6)])
    core.tick(cursor)
    core.plant.wait()
    errs.append(np.linalg.norm(core.target - core.tcp))
    notes.add(core.note)
    assert core.target[1] <= core.targets.hi[1] + 1e-9 and core.target[2] >= core.targets.lo[2] - 1e-9
  print(f"[test] 12 s figure-eight: tracking error mean {np.mean(errs) * 1e3:.1f} mm, max target y {core.targets.hi[1]:+.3f} respected, "
        f"min z respected, notes {sorted(n for n in notes if n)}; message: {core.message}")


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--port", default=FOLLOWER_PORT)
  p.add_argument("--speed", type=float, default=0.25, help="fraction of the full speed limit (0.05-1.0); +/- in the window")
  p.add_argument("--torque-limit", type=int, default=400)
  p.add_argument("--policy-dir", type=Path, default=POLICY)
  p.add_argument("--sim-bus", action="store_true")
  p.add_argument("--headless-test", action="store_true")
  args = p.parse_args()
  if not 0.05 <= args.speed <= 1.0:
    p.error("--speed must be between 0.05 and 1.0")
  jmap, calib = JointMap.load(), load_calibration()
  ctrl = ReachController(ControllerCfg(speed_factor=args.speed), policy_dir=args.policy_dir)
  targets = SafeTargets()
  if args.sim_bus:
    bus = SimServoBus(jmap, np.array([so101.HOME_JOINT_POS[n] for n in so101.ARM_JOINTS]))
    plant = SimRealPlant(bus, jmap, calib, torque_limit=args.torque_limit)
    plant.engage()
    core = MouseFollowCore(plant, ctrl, targets)
    headless_test(core) if args.headless_test else run_window(core)
    return
  with FeetechBus(args.port, calib=calib) as bus:
    plant = RealPlant(bus, jmap, calib, torque_limit=args.torque_limit)
    plant.engage()
    print("[mouse] torque ON, holding. Window: left click / F to follow, Esc to hold + quit.")
    core = MouseFollowCore(plant, ctrl, targets)
    run_window(core)
    print("[mouse] exited; arm holding pose (torque on). scripts/real_release.py lets it go limp.")


if __name__ == "__main__":
  main()
