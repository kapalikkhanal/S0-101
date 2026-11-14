"""Read-only hardware inspection before any motion: never writes a servo register.

  .venv/bin/python scripts/real_readonly_check.py --names "white cube" "white box"

Reads joint positions/torque state/temperatures once, then uses only the cameras:
head detections with one-to-one assignment and measured geometry, the projected arm
silhouette and the obstacle cloud the motion monitor would see, the wrist view with the
model's jaw tips projected into it, and a short run of the real camera worker.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--names", nargs="+", default=["white cube", "white box"])
  p.add_argument("--worker-seconds", type=float, default=8.0)
  p.add_argument("--out", type=Path, default=ROOT / "results" / f"hardware_{datetime.now():%Y-%m-%d}" / f"readonly_{datetime.now():%H%M%S}")
  args = p.parse_args()
  args.out.mkdir(parents=True, exist_ok=True)

  from scipy.spatial import cKDTree
  from so101_reach.agent.perception import RealPerception, View
  from so101_reach.manipulation.scene import CameraScene, RobotSurface, assign_detections
  from so101_reach.real.cameras import open_camera
  from so101_reach.real.feetech_bus import JOINT_ORDER, FeetechBus, load_calibration
  from so101_reach.real.grasp_replay import ArmKinematics
  from so101_reach.real.handeye import HeadCameraCalibration, WristCameraCalibration, WristCameraPose
  from so101_reach.real.joint_map import FOLLOWER_PORT, JointMap
  from so101_reach.real.orange import OrangeDetector

  report = {"names": args.names}
  # 1. Servo bus: reads only.
  jmap, fcal = JointMap.load(), load_calibration()
  with FeetechBus(FOLLOWER_PORT, calib=fcal) as bus:
    raw = bus.sync_read("Present_Position")
    torque = bus.sync_read("Torque_Enable")
    temps = {n: bus.read(n, "Present_Temperature") for n in JOINT_ORDER}
    volts = {n: bus.read(n, "Present_Voltage") / 10 for n in JOINT_ORDER}
  q = np.array([jmap.to_rad(n, raw[n]) for n in JOINT_ORDER])
  report["servo"] = {n: {"ticks": int(raw[n]), "model_deg": round(float(np.degrees(q[i])), 1), "torque": int(torque[n]),
                         "temp_C": temps[n], "volt": volts[n], "in_calibrated_range": bool(fcal[n].range_min <= raw[n] <= fcal[n].range_max)}
                     for i, n in enumerate(JOINT_ORDER)}
  print("servos:", json.dumps(report["servo"]), flush=True)

  kin = ArmKinematics()
  surface = RobotSurface(kin.m)
  kin.fk(q)
  robot_pts = surface.world(kin.d)
  jaw_ids = [kin.m.geom(f"{s}_jaw_sph_tip{i}").id for s in ("fixed", "moving") for i in (1, 2, 3)]
  jaw_tips = kin.d.geom_xpos[jaw_ids].copy()

  with contextlib.ExitStack() as stack:
    # 2. Head camera: detections, geometry, arm silhouette, obstacle cloud.
    head = stack.enter_context(open_camera("head"))
    head.warmup(1.5)
    fr = head.read()
    hcal = HeadCameraCalibration.load()
    tz = OrangeDetector(head, hcal).estimate_table_z(fr)
    report["table_z_m"] = tz
    print("table z:", tz, flush=True)
    if tz is None:
      raise SystemExit("cannot measure table height")
    per = RealPerception(head, None, hcal, None, tz, memory_frames=0)
    per.arm_surface = lambda: robot_pts
    per.frames = {"head": fr}
    R, t = hcal.pose()
    k = head.intrinsics
    view = View(fr.color, R, t, k.fx, k.fy, k.cx, k.cy)
    t0 = time.perf_counter()
    cands = {n: per.find(n, keep_ratio=0.0, frame=fr, remember=False) for n in args.names}
    report["detect_s"] = time.perf_counter() - t0
    chosen, info = assign_detections(cands)
    report["assignment"] = info
    report["candidates"] = {n: [{"score": round(float(d.score), 3), "center_m": np.round(d.center, 4).tolist(),
                                 "top_above_table_mm": round(1000 * (d.top_z - tz), 1), "size_mm": [round(1000 * s, 1) for s in d.size],
                                 "yaw_deg": round(d.yaw_deg, 1), "bbox": d.bbox.astype(int).tolist()} for d in ds] for n, ds in cands.items()}
    print("assignment:", info, flush=True)
    for n, ds in report["candidates"].items():
      print(f"  {n}: {ds}", flush=True)
    img = fr.color.copy()
    mask = per._arm_mask(view, img.shape[:2])
    img[mask] = (0.5 * img[mask] + [0, 0, 127]).astype(np.uint8)
    depth = np.where(mask, 0, fr.depth)
    vv, uu = np.mgrid[0:depth.shape[0]:6, 0:depth.shape[1]:6]
    zz = depth[vv, uu]
    ok = (zz > 0.08) & (zz < 1.8)
    pc = np.column_stack(((uu[ok] - k.cx) / k.fx * zz[ok], (vv[ok] - k.cy) / k.fy * zz[ok], zz[ok])) @ R.T + t
    keep = (pc[:, 2] > tz + 0.004) & (np.linalg.norm(pc[:, :2], axis=1) < 0.60)
    pc, px_u, px_v = pc[keep], uu[ok][keep], vv[ok][keep]
    near_robot = np.isfinite(cKDTree(robot_pts).query(pc, distance_upper_bound=0.015)[0])
    report["cloud_points_total"] = int(len(pc))
    report["cloud_points_self_filtered"] = int(near_robot.sum())
    for u, v, s in zip(px_u, px_v, near_robot):
      cv2.circle(img, (int(u), int(v)), 1, (255, 255, 0) if s else (0, 165, 255), -1)
    for n, ds in cands.items():
      for d in ds:
        b = d.bbox.astype(int)
        cv2.rectangle(img, tuple(b[:2]), tuple(b[2:]), (180, 180, 180), 1)
    for n, d in chosen.items():
      b = d.bbox.astype(int)
      cv2.rectangle(img, tuple(b[:2]), tuple(b[2:]), (0, 220, 0), 2)
      cv2.putText(img, f"{n} {d.score:.2f} top {1000 * (d.top_z - tz):.0f}mm {1000 * d.size[0]:.0f}x{1000 * d.size[1]:.0f}",
                  (b[0], max(15, b[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)
    for p3 in jaw_tips:
      pp = view.project(p3)
      if pp is not None:
        cv2.circle(img, pp, 4, (255, 0, 255), -1)
    cv2.imwrite(str(args.out / "head_check.jpg"), img)
    cv2.imwrite(str(args.out / "head_raw.jpg"), fr.color)

    # 3. Wrist camera: raw view, projected jaw tips, detections.
    wrist = stack.enter_context(open_camera("wrist"))
    wrist.warmup(1.0)
    wf = wrist.read()
    wpose = WristCameraPose(WristCameraCalibration.load(), lambda: {n: float(q[i]) for i, n in enumerate(JOINT_ORDER)})
    Rw, tw = wpose.pose()
    wk = wrist.intrinsics
    wview = View(wf.color, Rw, tw, wk.fx, wk.fy, wk.cx, wk.cy)
    wimg = wf.color.copy()
    for p3 in jaw_tips:
      pp = wview.project(p3)
      if pp is not None:
        cv2.circle(wimg, pp, 5, (255, 0, 255), -1)
    wdets = {}
    for n in args.names:
      res = per.model.detect(wf.color, n, 0.3, 0.3)
      wdets[n] = []
      for score, box in res:
        b = box.astype(int)
        patch = wf.depth[max(0, b[1]):b[3], max(0, b[0]):b[2]]
        valid = patch[patch > 0]
        wdets[n].append({"score": round(float(score), 3), "bbox": b.tolist(), "valid_depth_frac": round(float(valid.size / max(patch.size, 1)), 2),
                         "median_depth_m": round(float(np.median(valid)), 3) if valid.size else None})
        cv2.rectangle(wimg, tuple(b[:2]), tuple(b[2:]), (0, 220, 0), 2)
        cv2.putText(wimg, f"{n} {score:.2f}", (b[0], max(15, b[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)
    report["wrist"] = {"detections": wdets, "valid_depth_frac_image": round(float((wf.depth > 0).mean()), 2),
                       "jaw_tips_in_image": [wview.project(p3) for p3 in jaw_tips]}
    print("wrist:", json.dumps(report["wrist"]), flush=True)
    cv2.imwrite(str(args.out / "wrist_check.jpg"), wimg)
    cv2.imwrite(str(args.out / "wrist_raw.jpg"), wf.color)
    wrist.close()
    stack.pop_all()  # cameras closed explicitly below
    # 4. The real camera worker on the static scene (arm not moving).
    per2 = RealPerception(head, None, hcal, None, tz, memory_frames=0)
    scene = CameraScene(per2, args.names)
    scene.set_joints(q)
    end = time.monotonic() + args.worker_seconds
    ages = []
    while time.monotonic() < end:
      s = scene.snapshot()
      if s.error:
        report["worker_error"] = s.error
        break
      if s.stamp:
        ages.append(time.monotonic() - s.stamp)
      time.sleep(0.05)
    s = scene.snapshot()
    scene.close()
    head.close()
    ls = np.array(scene.loop_s or [np.nan])
    report["worker"] = {"loops": len(scene.loop_s), "loop_s_median": float(np.median(ls)), "loop_s_max": float(ls.max()),
                        "snapshot_age_median_s": float(np.median(ages)) if ages else None, "snapshot_age_max_s": float(max(ages)) if ages else None,
                        "fresh_at_end": {n: bool(n in s.seen and time.monotonic() - s.seen[n] < 0.8) for n in args.names},
                        "assignment": scene.assignment, "timing_median_s": {k2: float(np.median(v)) for k2, v in scene.timing.items() if v},
                        "timeline": scene.timeline[-10:], "cloud_points": int(len(s.cloud)),
                        "objects": {n: {"center_m": np.round(d.center, 4).tolist(), "top_above_table_mm": round(1000 * (d.top_z - tz), 1),
                                        "size_mm": [round(1000 * x, 1) for x in d.size], "yaw_deg": round(d.yaw_deg, 1)} for n, d in s.objects.items()}}
    print("worker:", json.dumps({k2: v for k2, v in report["worker"].items() if k2 != "timeline"}), flush=True)
  (args.out / "report.json").write_text(json.dumps(report, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)) + "\n")
  print("saved", args.out, flush=True)


if __name__ == "__main__":
  main()
