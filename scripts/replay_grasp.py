"""Replay the taught orange grasp at a new orange position (MOVES THE FOLLOWER).

  python scripts/replay_grasp.py                 # window: hover above the orange, G = grasp
  python scripts/replay_grasp.py --auto          # grasp by itself once settled above the orange
  python scripts/replay_grasp.py --put-down      # no cameras: put the held object down where it
                                                 # is, open, rise 6 cm, go home
  python scripts/replay_grasp.py --object "red cube" --place "white box" --trials 5
      # pick the red cube and drop it into the white box (both found by text)
  python scripts/replay_grasp.py --object apple --method orange --trials 10
      # any object by text (Grounding DINO), picked with the grasp taught as "orange"
  python scripts/replay_grasp.py --trials 10 --replay-speed 1.0 --speed 0.25
      # 10 pick trials: hover, grasp, hold 2 s, judge (gripper stopped on the orange?), put down,
      # home; then move the orange and press SPACE for the next trial. Y / N correct the last
      # verdict. Results: data/grasp_trials/trials_<time>.json

Speeds: --speed is the reach policy's speed while hovering (0.05-1.0, +/- keys change it live);
--replay-speed scales the grasp, put-down and home motions vs the taught demo (0.25-2.0).

Window keys: F / click follow the orange, G grasp (replays the newest data/grasp_demos demo,
retargeted to the orange), P place (into the --place container, else put it back down) + home,
D put it back down where it was picked + home, H home, SPACE next trial, Y / N fix the last
trial's verdict, Esc quit (arm holds).
--place: the container is found and measured once at the start (head camera, arm at home);
objects already inside it are ignored. The start is refused if it cannot be reached.
The grasp is refused if the retargeted path is not reachable or comes within 1 cm of the table.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import time
from datetime import datetime
from pathlib import Path

os.environ["MUJOCO_GL"] = "glfw"

import glfw  # noqa: E402
import numpy as np  # noqa: E402

from so101_reach.real.feetech_bus import JOINT_ORDER, FeetechBus, load_calibration  # noqa: E402
from so101_reach.real.grasp_replay import (  # noqa: E402
  ArmKinematics,
  GraspDemo,
  container_sequence,
  GraspReplaySession,
  execute,
  latest_demo,
  list_methods,
  place_sequence,
)
from so101_reach.real.joint_map import FOLLOWER_PORT, JointMap  # noqa: E402
from so101_reach.real.robot import RealPlant  # noqa: E402
from so101_reach.sim import so101  # noqa: E402

from so101_reach.settings import POLICY_DIR as POLICY


def place_only(demo: GraspDemo, countdown: float) -> None:
  jmap, fcal = JointMap.load(), load_calibration()
  kin = ArmKinematics()
  with FeetechBus(FOLLOWER_PORT, calib=fcal) as bus:
    plant = RealPlant(bus, jmap, fcal, torque_limit=400)
    plant.engage(keep_gripper=True)
    raw = bus.sync_read("Present_Position")
    q6 = np.array([jmap.to_rad(n, raw[n]) for n in JOINT_ORDER])
    tcp, _ = kin.fk(q6)
    seq = place_sequence(kin, jmap, demo, q6, raw["gripper"] - 30, demo.orange[2])
    print(f"[place] TCP now ({tcp[0]:.3f}, {tcp[1]:.3f}, {tcp[2]:.3f}); lowering to z {seq[0][1].targets[-1, 2]:.3f}, "
          f"opening, rising, home (IK error max {max(p.pos_err.max() for _, p in seq) * 1e3:.1f} mm)", flush=True)
    for k in range(int(countdown), 0, -1):
      print(f"[place] moving in {k}...", flush=True)
      time.sleep(1.0)
    for name, plan in seq:
      st = execute(bus, plan, jmap, fcal, approach_s=0.3, gripper_torque=250)
      print(f"[place] {name}: {st}", flush=True)
      if st != "ok":
        print("[place] stopped - the arm holds its pose")
        return
  print("[place] done: orange on the table, arm home")


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--demo", type=Path, default=None, help="demo .npz (default: newest)")
  p.add_argument("--speed", type=float, default=0.25, help="reach-policy speed while hovering")
  p.add_argument("--replay-speed", type=float, default=1.0, help="grasp / put-down / home speed vs the demo (0.25-2)")
  p.add_argument("--trials", type=int, default=0, help="run N pick trials (implies --auto)")
  p.add_argument("--hold", type=float, default=2.0, help="seconds to hold the lifted orange in a trial")
  p.add_argument("--hover", type=float, default=0.05)
  p.add_argument("--auto", action="store_true")
  p.add_argument("--put-down", "--place-only", dest="place_only", action="store_true",
                 help="no cameras: put the held object down where it is, open, go home")
  p.add_argument("--min-score", type=float, default=0.4,
                 help="Grounding DINO confidence needed for --object (lower it if the object is missed, e.g. 0.3)")
  p.add_argument("--place", default=None,
                 help='container to drop the picked object into, by text, e.g. "white box", "brown box"')
  p.add_argument("--countdown", type=float, default=3.0)
  p.add_argument("--no-wrist", action="store_true")
  p.add_argument("--object", default=None,
                 help='text prompt for the Grounding DINO detector, e.g. "apple", "peach", "red ball" (default: orange colour detector)')
  p.add_argument("--method", default=None,
                 help="taught grasp to replay, e.g. orange (default: the object's own method if taught, else orange)")
  p.add_argument("--policy-dir", type=Path, default=POLICY)
  args = p.parse_args()
  if not 0.25 <= args.replay_speed <= 2.0:
    p.error("--replay-speed must be within 0.25-2.0")
  if not 0.05 <= args.speed <= 1.0:
    p.error("--speed must be within 0.05-1.0")
  methods = list_methods()
  method = args.method or (args.object if args.object in methods else "orange")
  demo_path = args.demo or latest_demo(method)
  demo = GraspDemo.load(demo_path)
  print(f"[replay] object {args.object or 'orange'!r} (detector: {'Grounding DINO' if args.object else 'orange colour'}), "
        f"method {demo.meta.get('method', 'orange')!r}: demo {demo_path.name}, {len(demo.t)} samples, {demo.t[-1]:.1f} s; "
        f"taught methods: {', '.join(methods)}", flush=True)
  if args.place_only:
    place_only(demo, args.countdown)
    return

  from so101_reach.real.cameras import open_camera
  from so101_reach.real.controller import ControllerCfg, ReachController
  from so101_reach.real.handeye import WRIST_EXTRINSICS_PATH, HeadCameraCalibration, WristCameraCalibration, WristCameraPose
  from so101_reach.real.mouse_real import MouseFollowCore, SafeTargets
  from so101_reach.real.containers import draw_container, find_container
  from so101_reach.real.objects import make_detector
  from so101_reach.real.orange_follow import OrangeTracker, run_window

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
    dkw = {"box_threshold": args.min_score} if args.object else {}
    det = make_detector(head, hcal, args.object, **dkw)
    det.table_z = det.estimate_table_z(head.read())
    if det.table_z is None:
      raise SystemExit("cannot measure the table height from the head camera")
    sources = [("head", head, det)]
    if not args.no_wrist and WRIST_EXTRINSICS_PATH.exists():
      wrist = stack.enter_context(open_camera("wrist"))
      wrist.warmup(1.0)
      sources.append(("wrist", wrist, make_detector(wrist, WristCameraPose(WristCameraCalibration.load(), joints), args.object, table_z=det.table_z, **dkw)))
    name = args.object or "orange"
    tr = OrangeTracker(sources, hover=args.hover, name=name, **({"fresh": 0.4, "lost_after": 0.8} if args.object else {}))
    container = None
    if args.place:
      container = find_container(head, hcal, det.table_z, args.place)
      if container is None:
        raise SystemExit(f"[place] no {args.place!r} found by the head camera")
      print(f"[place] {container.describe()}", flush=True)
      # Reach check before anything moves: carry the demo's lifted pose into it.
      kin = ArmKinematics()
      try:
        container_sequence(kin, jmap, container, demo.q[-1], 2000, float(demo.gripper_goal.max()), 0.026,
                           SafeTargets().lo, SafeTargets().hi)
      except ValueError as e:
        raise SystemExit(f"[place] cannot place into the {args.place}: {e} - move it closer to the arm") from None
      drop = container.drop_point(kin.pan_anchor, 0.026 + 0.015)
      tr.overlay = lambda img: draw_container(img, container, head, hcal, drop)
      for _, _, d in sources:
        d.exclude = [container]

    bus = stack.enter_context(FeetechBus(FOLLOWER_PORT, calib=fcal))
    plant = RealPlant(bus, jmap, fcal, torque_limit=400)
    plant.engage()
    core = MouseFollowCore(plant, ctrl, SafeTargets(), source=f"the {name}")
    holder["core"] = core
    log_path = None
    if args.trials:
      log_path = so101.PROJECT_ROOT / "data" / "grasp_trials" / f"trials_{datetime.now():%Y%m%d_%H%M%S}.json"
      print(f"[trial] {args.trials} trials, hover speed {args.speed}, replay speed {args.replay_speed}; log {log_path}", flush=True)
    sess = GraspReplaySession(core, tr, demo, fcal, speed=args.replay_speed, auto=args.auto, trials=args.trials,
                              hold_s=args.hold, log_path=log_path, container=container)
    if args.auto or args.trials:
      core.toggle()
    tr.start()
    try:
      run_window(
        core,
        tr,
        step_fn=sess.step,
        keys={glfw.KEY_G: sess.press_g, glfw.KEY_P: sess.press_p, glfw.KEY_D: sess.press_d, glfw.KEY_H: sess.press_h,
              glfw.KEY_SPACE: sess.press_space,
              glfw.KEY_Y: lambda: sess.mark(True), glfw.KEY_N: lambda: sess.mark(False),
              "F": lambda: core.toggle() if sess.state == "FOLLOW" else None},
        extra_status=sess.status,
        title="SO-101 PICK & PLACE - F follow, G grasp, P place, D put back, H home, Esc quit",
        hold_on_exit=False,
      )
    finally:
      tr.stop.set()
      if sess.player is not None and not sess.player.done:
        sess.player._hold("aborted")
      elif sess.state == "FOLLOW":
        plant.hold()
      time.sleep(0.2)
  if sess.results:
    n_ok = sum(r["picked"] for r in sess.results)
    print(f"[trial] RESULT: {n_ok}/{len(sess.results)} picked; log {log_path}")
  print(f"[replay] exited in state {sess.state}; the arm holds its pose.")


if __name__ == "__main__":
  main()
