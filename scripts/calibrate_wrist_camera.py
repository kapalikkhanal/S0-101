"""Calibrate the wrist camera (D405) mount on the gripper, using the orange as the reference
(MOVES THE ARM).

1. The arm goes home so the head camera can see the orange; its centre is measured there
   (head camera + calibration; the orange must stay still for the whole run).
2. The arm hovers at a grid of points behind (toward the base), beside and above the orange -
   directly above it the gripper jaws hide the orange from the wrist camera; at each, the wrist camera
   measures the viewing ray + surface depth to the orange and the joints give the gripper pose.
3. Solve the camera pose in the gripper frame (+ depth offset) -> configs/wrist_camera_extrinsics.json

  python scripts/calibrate_wrist_camera.py --manual        # hand-guided (recommended)
  python scripts/calibrate_wrist_camera.py --speed 0.25    # automatic grid (MOVES THE ARM)

Hand-guided: torque off; first move the arm away so the head camera sees the orange (Enter),
then pose the arm so the wrist camera sees it (Enter per pose, 12-15 poses), q to solve.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np

from so101_reach.real.cameras import open_camera
from so101_reach.real.controller import ControllerCfg, ReachController
from so101_reach.real.feetech_bus import FeetechBus, load_calibration
from so101_reach.real.handeye import HeadCameraCalibration, gripper_pose, solve_wrist
from so101_reach.real.joint_map import FOLLOWER_PORT, JointMap
from so101_reach.real.mouse_real import SafeTargets
from so101_reach.real.orange import OrangeDetector
from so101_reach.real.robot import RealPlant, SafetyStop, run_live
from so101_reach.sim import so101

from so101_reach.settings import POLICY_DIR as POLICY
HOME_TCP = np.array([0.236, 0.0, 0.132])


class _Identity:
  def pose(self):
    return np.eye(3), np.zeros(3)


def head_orange(cam, det, seconds: float = 1.5) -> np.ndarray | None:
  pts = []
  t0 = time.time()
  while time.time() - t0 < seconds:
    o = det.detect(cam.read())
    if o is not None:
      pts.append(o.center_base)
  return np.median(pts, axis=0) if len(pts) >= 10 else None


def wrist_sighting(cam, det, plant, seconds: float):
  rays, deps, qs = [], [], []
  t0 = time.time()
  while time.time() - t0 < seconds:
    qs.append({**dict(zip(so101.ARM_JOINTS, plant.read_q(), strict=True)), **plant.fixed})
    o = det.detect(cam.read())
    if o is None or not np.isfinite(o.surface_depth):
      continue
    r = cam.deproject(*o.pixel, 1.0)
    rays.append(r / np.linalg.norm(r))
    deps.append(o.surface_depth)
  if len(rays) < 8:
    return None
  ray = np.median(rays, axis=0)
  q = {n: float(np.mean([x[n] for x in qs])) for n in qs[0]}
  return ray / np.linalg.norm(ray), float(np.median(deps)), q, len(rays)


def manual(head_cal, measure_s: float):
  import queue
  import threading

  jmap, calib = JointMap.load(), load_calibration()
  model = so101.make_cpu_model()
  data = mujoco.MjData(model)
  rows = []
  with open_camera("head") as head, open_camera("wrist") as wrist, FeetechBus(FOLLOWER_PORT, calib=calib) as bus:
    head.warmup(1.0)
    wrist.warmup(1.0)
    hdet = OrangeDetector(head, head_cal)
    wdet = OrangeDetector(wrist, _Identity(), min_area_px=500)
    plant = RealPlant(bus, jmap, calib)
    if any(bus.sync_read("Torque_Enable").values()):
      print("[wrist-calib] SUPPORT THE ARM - torque goes off in:")
      for k in range(5, 0, -1):
        print(f"  {k}...", flush=True)
        time.sleep(1.0)
      plant.release()
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
    orange = None
    print("[wrist-calib] Step 1: move the arm away so the HEAD camera sees the orange, then press Enter.")
    last = 0.0
    while True:
      wfr = wrist.read()
      wvis = wdet.detect(wfr) is not None
      hfr = head.read()
      if hdet.table_z is None:
        hdet.table_z = hdet.estimate_table_z(hfr)
      hvis = hdet.table_z is not None and hdet.detect(hfr) is not None
      if time.time() - last > 0.5:
        last = time.time()
        stage = "head" if orange is None else "wrist"
        print(f"\r[wrist-calib] head camera: {'VISIBLE' if hvis else 'not visible'} | wrist camera: {'VISIBLE' if wvis else 'not visible'}"
              f" | recorded {len(rows)} (now: {stage})   ", end="", flush=True)
      if cmd.empty():
        continue
      c = cmd.get()
      if c == "q":
        print()
        break
      if orange is None:
        orange = head_orange(head, hdet)
        if orange is None:
          print("\n[wrist-calib] head camera does not see the orange well - move the arm further away and retry")
          continue
        print(f"\n[wrist-calib] orange at {np.round(orange, 4)} m (table z {hdet.table_z * 1e3:+.1f} mm). Step 2: pose the arm so the "
              "WRIST camera sees the orange, Enter per pose (12-15 poses, vary distance/height/tilt), q to solve.")
        continue
      if hvis:
        now = head_orange(head, hdet, 0.5)
        if now is not None and np.linalg.norm(now - orange) > 0.01:
          print(f"\n[wrist-calib] WARNING: the orange moved {np.linalg.norm(now - orange) * 1e3:.0f} mm - restart the calibration")
      s = wrist_sighting(wrist, wdet, plant, measure_s)
      if s is None:
        print("\n[wrist-calib] wrist camera did not see the orange well enough - adjust and retry")
        continue
      ray, depth, q, n = s
      R, t = gripper_pose(model, data, q)
      rows.append((ray, depth, R, t))
      print(f"\n[wrist-calib] pose {len(rows)}: orange {depth * 100:.1f} cm from the wrist camera ({n} reads)")
  return rows, orange


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--speed", type=float, default=0.25)
  p.add_argument("--countdown", type=float, default=10.0)
  p.add_argument("--policy-dir", type=Path, default=POLICY)
  p.add_argument("--dx", type=float, nargs="+", default=[-0.11, -0.08, -0.05], help="gripper offsets from the orange along x [m]")
  p.add_argument("--dy", type=float, nargs="+", default=[-0.05, 0.0, 0.05])
  p.add_argument("--dz", type=float, nargs="+", default=[0.09, 0.14])
  p.add_argument("--manual", action="store_true", help="hand-guided: torque off, Enter records each pose")
  args = p.parse_args()

  head_cal = HeadCameraCalibration.load()
  if args.manual:
    rows, orange = manual(head_cal, 1.0)
    finish(rows, orange)
    return
  jmap, calib = JointMap.load(), load_calibration()
  ctrl = ReachController(ControllerCfg(speed_factor=args.speed), policy_dir=args.policy_dir)
  safe = SafeTargets(min_z=0.09)
  model = so101.make_cpu_model()
  data = mujoco.MjData(model)
  rows = []
  with open_camera("head") as head, open_camera("wrist") as wrist, FeetechBus(FOLLOWER_PORT, calib=calib) as bus:
    head.warmup(1.0)
    wrist.warmup(1.0)
    hdet = OrangeDetector(head, head_cal)
    wdet = OrangeDetector(wrist, _Identity(), min_area_px=500)
    plant = RealPlant(bus, jmap, calib, torque_limit=400)
    for k in range(int(args.countdown), 0, -1):
      print(f"[wrist-calib] moving in {k} s - do not touch the orange", flush=True)
      time.sleep(1.0)
    plant.engage()
    try:
      run_live(plant, ctrl, HOME_TCP, duration=8.0)
      plant.hold()
      time.sleep(0.5)
      hdet.table_z = hdet.estimate_table_z(head.read())
      if hdet.table_z is None:
        raise SystemExit("[wrist-calib] could not measure the table height from the head camera")
      orange = head_orange(head, hdet)
      if orange is None:
        raise SystemExit("[wrist-calib] head camera cannot see the orange from home - place it in view and retry")
      print(f"[wrist-calib] orange (head camera): {np.round(orange, 4)} m; table z {hdet.table_z * 1e3:+.1f} mm", flush=True)
      if not (-0.03 < hdet.table_z < 0.01 and 0.0 < orange[2] - hdet.table_z < 0.06):
        raise SystemExit("[wrist-calib] implausible table/orange height - not moving")
      targets = []
      for dz in args.dz:
        for k, dx in enumerate(args.dx):
          for dy in (args.dy if k % 2 == 0 else args.dy[::-1]):
            t, note = safe(orange + np.array([dx, dy, dz]))
            if not note:
              targets.append(t)
      print(f"[wrist-calib] {len(targets)} poses around the orange", flush=True)
      for i, tgt in enumerate(targets):
        try:
          run_live(plant, ctrl, tgt, duration=8.0)
        except SafetyStop as e:
          plant.hold()
          print(f"[wrist-calib] {i + 1}: SAFETY STOP ({e}); skipped", flush=True)
          continue
        plant.hold()
        time.sleep(0.5)
        s = wrist_sighting(wrist, wdet, plant, 1.0)
        if s is None:
          print(f"[wrist-calib] {i + 1}/{len(targets)} {np.round(tgt, 3)}: orange not in wrist view - skipped", flush=True)
          continue
        ray, depth, q, n = s
        R, t = gripper_pose(model, data, q)
        rows.append((ray, depth, R, t))
        print(f"[wrist-calib] {i + 1}/{len(targets)} {np.round(tgt, 3)}: wrist sees orange at {depth * 100:.1f} cm ({n} reads)", flush=True)
      run_live(plant, ctrl, HOME_TCP, duration=8.0)
    finally:
      plant.hold()
  finish(rows, orange)


def finish(rows: list, orange: np.ndarray | None) -> None:
  if len(rows) < 6 or orange is None:
    raise SystemExit(f"[wrist-calib] only {len(rows)} sightings - not enough")
  rays = np.array([r[0] for r in rows])
  deps = np.array([r[1] for r in rows])
  Rs = np.array([r[2] for r in rows])
  Ts = np.array([r[3] for r in rows])
  out = so101.PROJECT_ROOT / "logs" / "real" / f"wrist_calib_{datetime.now():%Y%m%d_%H%M%S}.npz"
  np.savez(out, rays=rays, depths=deps, R_base_gripper=Rs, t_base_gripper=Ts, orange=orange)
  cal, res = solve_wrist(rays, deps, Rs, Ts, orange)
  cal.save()
  print(f"[wrist-calib] {cal.num_poses} sightings: rms {cal.rms_mm:.1f} mm, leave-one-out {cal.loo_rms_mm:.1f} mm, max {res.max() * 1e3:.1f} mm")
  print(f"[wrist-calib] camera in gripper frame {np.round(cal.t_gripper_cam, 4)} m, depth offset {cal.depth_offset * 1e3:.1f} mm")
  print(f"[wrist-calib] saved configs/wrist_camera_extrinsics.json (raw {out})")


if __name__ == "__main__":
  main()
