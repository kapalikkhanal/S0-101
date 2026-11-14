"""Calibrate the fixed head camera (D435i) against the robot base.

Automatic mode MOVES THE ARM: it visits a grid of points in front of it (fine-tuned reach policy, reduced speed,
site limits); at each point the tag centre is measured in 3-D by the head camera (colour
pixel + aligned depth) and the gripper pose by forward kinematics of the measured joints.
Result: configs/head_camera_extrinsics.json (+ raw data in logs/real/).

  python scripts/calibrate_head_camera.py --manual      # hand-guided (recommended): torque off,
                                                        # pose the arm, Enter records, q solves
  python scripts/calibrate_head_camera.py --speed 0.25  # automatic grid (the tag must stay visible)
"""

from __future__ import annotations

import argparse
import itertools
import time
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np

from so101_reach.real.cameras import open_camera
from so101_reach.real.controller import ControllerCfg, ReachController
from so101_reach.real.feetech_bus import FeetechBus, load_calibration
from so101_reach.real.handeye import gripper_pose, solve
from so101_reach.real.joint_map import FOLLOWER_PORT, JointMap
from so101_reach.real.mouse_real import SafeTargets
from so101_reach.real.robot import RealPlant, SafetyStop, run_live
from so101_reach.real.tags import TagDetector
from so101_reach.sim import so101

from so101_reach.settings import POLICY_DIR as POLICY


def grid() -> list[np.ndarray]:
  xs, ys, zs = (0.17, 0.23, 0.29), (-0.12, -0.04, 0.04), (0.10, 0.18)
  pts = []
  for k, (z, x) in enumerate(itertools.product(zs, xs)):
    for y in (ys if k % 2 == 0 else ys[::-1]):  # serpentine
      pts.append(np.array([x, y, z]))
  return pts


def measure(cam, det, plant, seconds: float) -> tuple[np.ndarray | None, dict[str, float], int]:
  pts, qs = [], []
  t0 = time.time()
  while time.time() - t0 < seconds:
    q = plant.read_q()
    qs.append({**dict(zip(so101.ARM_JOINTS, q, strict=True)), **plant.fixed})
    fr = cam.read()
    d = det.detect(fr.color)
    if d is None:
      continue
    z = cam.depth_at(fr.depth, *d.center)
    if np.isfinite(z):
      pts.append(cam.deproject(*d.center, z))
  qm = {n: float(np.mean([q[n] for q in qs])) for n in qs[0]}
  return (np.median(pts, axis=0) if len(pts) >= 5 else None), qm, len(pts)


def check_consistency(rows: list) -> list[int]:
  """Indices of poses whose tag moved inconsistently with the gripper (loose/detached tag):
  between pose pairs the tag displacement seen by the camera must match the gripper
  displacement from FK within 40 % + 15 mm."""
  bad = set()
  for i in range(1, len(rows)):
    dp = np.linalg.norm(rows[i][0] - rows[i - 1][0])
    dt = np.linalg.norm(rows[i][2] - rows[i - 1][2])
    if abs(dp - dt) > 0.4 * dt + 0.015:
      bad.add(i)
  return sorted(bad)


def manual(args) -> list:
  """Hand-guided collection: torque off, the user poses the arm, Enter records."""
  import queue
  import threading

  jmap, calib = JointMap.load(), load_calibration()
  det = TagDetector()
  model = so101.make_cpu_model()
  data = mujoco.MjData(model)
  rows = []
  with open_camera("head") as cam, FeetechBus(FOLLOWER_PORT, calib=calib) as bus:
    cam.warmup(1.5)
    plant = RealPlant(bus, jmap, calib)
    if any(bus.sync_read("Torque_Enable").values()):
      print("[calib] SUPPORT THE ARM - torque goes off in:")
      for k in range(5, 0, -1):
        print(f"  {k}...", flush=True)
        time.sleep(1.0)
      plant.release()
    print("[calib] Move the arm by hand so the tag faces the head camera; Enter = record, q = finish.")
    cmd: queue.Queue[str] = queue.Queue()

    def stdin_loop() -> None:
      while True:
        try:
          line = input().strip().lower()
        except EOFError:
          line = "q"
        cmd.put(line)
        if line == "q":
          return

    threading.Thread(target=stdin_loop, daemon=True).start()
    last = 0.0
    while True:
      fr = cam.read()
      d = det.detect(fr.color)
      if time.time() - last > 0.5:
        last = time.time()
        print(f"\r[calib] tag {'VISIBLE' if d is not None else 'not visible'} | recorded {len(rows)}   ", end="", flush=True)
      if not cmd.empty():
        c = cmd.get()
        if c == "q":
          print()
          break
        p_cam, q, n = measure(cam, det, plant, args.measure)
        if p_cam is None:
          print(f"\n[calib] tag not seen well enough ({n} reads) - try again")
          continue
        R, t = gripper_pose(model, data, q)
        rows.append((p_cam, R, t, t, q))
        print(f"\n[calib] pose {len(rows)}: tag in camera {np.round(p_cam, 4)} m, gripper at {np.round(t, 3)} m ({n} reads)")
  return rows


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--manual", action="store_true", help="hand-guided: torque off, Enter records each pose")
  p.add_argument("--speed", type=float, default=0.25)
  p.add_argument("--countdown", type=float, default=10.0)
  p.add_argument("--settle", type=float, default=0.5)
  p.add_argument("--measure", type=float, default=1.0)
  p.add_argument("--policy-dir", type=Path, default=POLICY)
  args = p.parse_args()
  rows = manual(args) if args.manual else automatic(args)
  bad = check_consistency(rows)
  if bad:
    print(f"[calib] WARNING: tag moved inconsistently with the gripper at poses {[b + 1 for b in bad]} (loose tag?) - dropped")
    rows = [r for i, r in enumerate(rows) if i not in bad]
  if len(rows) < 6:
    raise SystemExit(f"only {len(rows)} usable poses - not enough to calibrate")
  P = np.array([r[0] for r in rows])
  Rs = np.array([r[1] for r in rows])
  Ts = np.array([r[2] for r in rows])
  out = so101.PROJECT_ROOT / "logs" / "real" / f"head_calib_{datetime.now():%Y%m%d_%H%M%S}.npz"
  out.parent.mkdir(parents=True, exist_ok=True)
  np.savez(out, p_cam=P, R_base_gripper=Rs, t_base_gripper=Ts, targets=np.array([r[3] for r in rows]))
  cal, res = solve(P, Rs, Ts)
  cal.save()
  print(f"[calib] {cal.num_poses} poses: fit rms {cal.rms_mm:.1f} mm, leave-one-out rms {cal.loo_rms_mm:.1f} mm, max {res.max() * 1e3:.1f} mm")
  print(f"[calib] camera position in base frame {np.round(cal.t_base_cam, 3)} m; tag offset in gripper frame {np.round(cal.tag_offset_gripper, 4)} m")
  print(f"[calib] per-pose residuals [mm]: {np.round(res * 1e3, 1).tolist()}")
  print(f"[calib] saved {so101.PROJECT_ROOT / 'configs' / 'head_camera_extrinsics.json'} (raw data {out})")


def automatic(args) -> list:
  jmap, calib = JointMap.load(), load_calibration()
  ctrl = ReachController(ControllerCfg(speed_factor=args.speed), policy_dir=args.policy_dir)
  safe = SafeTargets(min_z=0.10)
  targets = []
  for t in grid():
    tt, note = safe(t)
    if note:
      print(f"[calib] skipping {np.round(t, 3)}: {note}")
    else:
      targets.append(tt)
  det = TagDetector()
  model = so101.make_cpu_model()
  data = mujoco.MjData(model)
  rows = []
  with open_camera("head") as cam, FeetechBus(FOLLOWER_PORT, calib=calib) as bus:
    cam.warmup(1.5)
    plant = RealPlant(bus, jmap, calib, torque_limit=400)
    print(f"[calib] {len(targets)} poses, speed {args.speed}")
    for k in range(int(args.countdown), 0, -1):
      print(f"[calib] moving in {k} s", flush=True)
      time.sleep(1.0)
    plant.engage()
    try:
      for i, tgt in enumerate(targets):
        try:
          run_live(plant, ctrl, tgt, duration=8.0)
        except SafetyStop as e:
          plant.hold()
          print(f"[calib] {i + 1}: SAFETY STOP ({e}); skipping pose")
          if e.record is not None:
            np.savez(so101.PROJECT_ROOT / 'logs' / 'real' / f'calib_stop_{datetime.now():%Y%m%d_%H%M%S}.npz', target=tgt, **e.record)
          continue
        plant.hold()
        time.sleep(args.settle)
        p_cam, q, n = measure(cam, det, plant, args.measure)
        if p_cam is None:
          print(f"[calib] {i + 1}/{len(targets)} {np.round(tgt, 3)}: tag not seen ({n} reads) - skipped", flush=True)
          continue
        R, t = gripper_pose(model, data, q)
        rows.append((p_cam, R, t, tgt, q))
        print(f"[calib] {i + 1}/{len(targets)} {np.round(tgt, 3)}: tag in camera {np.round(p_cam, 4)} m ({n} reads)", flush=True)
    except KeyboardInterrupt:
      print("[calib] interrupted")
    finally:
      plant.hold()
  return rows


if __name__ == "__main__":
  main()
