"""Teach the orange grasp once by teleoperation (MOVES BOTH ARMS).

  python scripts/teach_grasp.py            # 25 % speed, hover 5 cm above the orange

1. The follower hovers above the orange (F / click toggles following, as in real_orange.py);
   the LEADER ARM MIRRORS THE FOLLOWER (low torque) so both arms match.
2. Press R: 5 s countdown - take hold of the leader arm.
3. The leader goes limp and the follower copies it (recording): open the gripper, go down,
   grasp the orange, lift ~5 cm.
4. Press R: recording stops and is saved to data/grasp_demos/; the follower holds and keeps
   gripping. Esc quits (both arms keep holding their pose; scripts/real_release.py [--leader]
   lets an arm go limp).
"""

from __future__ import annotations

import argparse
import contextlib
import os
import time
from pathlib import Path

os.environ["MUJOCO_GL"] = "glfw"

import glfw  # noqa: E402
import numpy as np  # noqa: E402

from so101_reach.real.cameras import open_camera  # noqa: E402
from so101_reach.real.controller import ControllerCfg, ReachController  # noqa: E402
from so101_reach.real.feetech_bus import FeetechBus, load_calibration  # noqa: E402
from so101_reach.real.grasp_teach import GraspTeachSession  # noqa: E402
from so101_reach.real.handeye import (  # noqa: E402
  WRIST_EXTRINSICS_PATH,
  HeadCameraCalibration,
  WristCameraCalibration,
  WristCameraPose,
)
from so101_reach.real.joint_map import FOLLOWER_PORT, JointMap  # noqa: E402
from so101_reach.real.mouse_real import MouseFollowCore, SafeTargets  # noqa: E402
from so101_reach.real.objects import make_detector  # noqa: E402
from so101_reach.real.orange_follow import OrangeTracker, run_window  # noqa: E402
from so101_reach.real.robot import RealPlant  # noqa: E402
from so101_reach.real.teleop import LeaderArm, open_leader  # noqa: E402
from so101_reach.sim import so101  # noqa: E402

from so101_reach.settings import POLICY_DIR as POLICY


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--speed", type=float, default=0.25)
  p.add_argument("--hover", type=float, default=0.05)
  p.add_argument("--countdown", type=float, default=5.0)
  p.add_argument("--no-wrist", action="store_true")
  p.add_argument("--object", default=None,
                 help='text prompt for the Grounding DINO detector, e.g. "apple", "peach", "red ball" (default: orange colour detector)')
  p.add_argument("--method", default=None, help="name to save this grasp as (default: the object, or 'orange')")
  p.add_argument("--policy-dir", type=Path, default=POLICY)
  args = p.parse_args()

  hcal = HeadCameraCalibration.load()
  jmap, fcal = JointMap.load(), load_calibration()
  ctrl = ReachController(ControllerCfg(speed_factor=args.speed), policy_dir=args.policy_dir)
  holder: dict = {}

  def joints() -> dict[str, float]:
    c = holder.get("core")
    q = c.q if c is not None else [so101.HOME_JOINT_POS[n] for n in so101.ARM_JOINTS]
    fixed = c.plant.fixed if c is not None and c.plant.fixed else so101.FIXED_JOINT_TARGETS
    return {**dict(zip(so101.ARM_JOINTS, q, strict=True)), **fixed}

  with contextlib.ExitStack() as stack:
    head = stack.enter_context(open_camera("head"))
    head.warmup(1.5)
    det = make_detector(head, hcal, args.object)
    det.table_z = det.estimate_table_z(head.read())
    if det.table_z is None:
      raise SystemExit("cannot measure the table height from the head camera")
    sources = [("head", head, det)]
    if not args.no_wrist and WRIST_EXTRINSICS_PATH.exists():
      wrist = stack.enter_context(open_camera("wrist"))
      wrist.warmup(1.0)
      sources.append(("wrist", wrist, make_detector(wrist, WristCameraPose(WristCameraCalibration.load(), joints), args.object, table_z=det.table_z)))
    name = args.object or "orange"
    tr = OrangeTracker(sources, hover=args.hover, name=name, **({"fresh": 0.4, "lost_after": 0.8} if args.object else {}))

    fbus = stack.enter_context(FeetechBus(FOLLOWER_PORT, calib=fcal))
    lbus, lcal = open_leader()
    stack.enter_context(lbus)
    plant = RealPlant(fbus, jmap, fcal, torque_limit=400)
    plant.engage()
    core = MouseFollowCore(plant, ctrl, SafeTargets(), source=f"the {name}")
    holder["core"] = core
    leader = LeaderArm(lbus)
    leader.engage()
    print("[teach] follower + leader torque ON. The leader will now MIRROR the follower.", flush=True)
    sess = GraspTeachSession(core, tr, leader, fcal, lcal, countdown=args.countdown, method=args.method or name, obj=name)
    print(f"[teach] object {name!r}, saving the grasp as method {sess.method!r}", flush=True)
    tr.start()
    try:
      run_window(
        core,
        tr,
        step_fn=lambda: sess.step(tr.hover_target()),
        keys={glfw.KEY_R: sess.press_r, "F": sess.press_f},
        extra_status=sess.status,
        title="SO-101 TEACH GRASP - F follow, R teleop/record, Esc quit",
        hold_on_exit=False,
      )
    finally:
      tr.stop.set()
      if sess.state == "TELEOP":
        sess.press_r()  # Save what was recorded.
      if sess.state == "DONE":
        pass  # _finish already holds the pose and keeps the gripper goal.
      else:
        plant.hold()
      if leader.engaged:
        pos = leader.read()
        lbus.sync_write("Goal_Position", pos)  # Leader keeps holding where it is.
      time.sleep(0.2)
  if sess.saved:
    print(f"[teach] saved {sess.saved}\n[teach] {sess.summary}")
  print("[teach] exited; both arms hold their pose. scripts/real_release.py (--leader) lets an arm go limp.")


if __name__ == "__main__":
  main()
