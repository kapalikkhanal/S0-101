"""Refine the wrist-camera pose from the jaw tips it sees (arm held still, gripper only moves).

  .venv/bin/python scripts/calibrate_wrist_jaws.py

The fixed jaw is rigid relative to the D405 and the moving jaw's tip depends only on the
opening, so the jaw tips are known points in the gripper frame. They are found in the colour
image (dark jaw silhouettes entering from the right; the tip is the silhouette's leftmost
point) at several openings, and a small rotation+translation correction to the existing
calibration is fitted (regularised towards it). Writes configs/wrist_camera_refined.json;
the original configs/wrist_camera_extrinsics.json is not modified.
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def jaw_tip_pixels(bgr: np.ndarray) -> dict[str, tuple[float, float]] | None:
  """Leftmost points of the two large dark silhouettes touching the right image edge:
  'upper' (moving jaw) and 'lower' (fixed jaw) in this camera mounting."""
  hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
  dark = cv2.morphologyEx((hsv[..., 2] < 55).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
  n, lab, st, _ = cv2.connectedComponentsWithStats(dark)
  w = dark.shape[1]
  blobs = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > 3000 and st[i, cv2.CC_STAT_LEFT] + st[i, cv2.CC_STAT_WIDTH] >= w - 5]
  if len(blobs) < 2:
    return None
  tips = []
  h = dark.shape[0]
  for i in blobs:
    ys, xs = np.where(lab == i)
    sel = xs <= xs.min() + 6
    tx, ty = float(np.median(xs[sel])), float(np.median(ys[sel]))
    if ty < 8 or ty > h - 8 or tx < 8:
      return None  # tip clipped by the image border: not a real tip
    tips.append((tx, ty, st[i, cv2.CC_STAT_AREA]))
  tips = sorted(sorted(tips, key=lambda t: -t[2])[:2], key=lambda t: t[1])
  return {"upper": tips[0][:2], "lower": tips[1][:2]}


def main():
  from so101_reach.agent.arm import AgentArm
  from so101_reach.real.cameras import open_camera
  from so101_reach.real.controller import ControllerCfg, ReachController
  from so101_reach.real.feetech_bus import JOINT_ORDER, FeetechBus, load_calibration
  from so101_reach.real.grasp_replay import ArmKinematics
  from so101_reach.real.handeye import WristCameraCalibration
  from so101_reach.real.joint_map import FOLLOWER_PORT, JointMap
  from so101_reach.real.mouse_real import SafeTargets
  from so101_reach.real.robot import RealPlant
  from so101_reach.sim import so101

  out = ROOT / "results" / f"hardware_{datetime.now():%Y-%m-%d}" / f"wrist_jaw_calib_{datetime.now():%H%M%S}"
  out.mkdir(parents=True, exist_ok=True)
  jmap, fcal = JointMap.load(), load_calibration()
  kin = ArmKinematics()
  m = kin.m
  gr = m.body("gripper").id
  tip_ids = {"lower": [m.geom(f"fixed_jaw_sph_tip{i}").id for i in (1, 2, 3)],
             "upper": [m.geom(f"moving_jaw_sph_tip{i}").id for i in (1, 2, 3)]}
  base = WristCameraCalibration.load()
  R0, t0 = np.array(base.R_gripper_cam), np.array(base.t_gripper_cam)
  obs = []  # (gripper-frame point, observed pixel)
  with FeetechBus(FOLLOWER_PORT, calib=fcal) as bus, open_camera("wrist") as cam:
    plant = RealPlant(bus, jmap, fcal, torque_limit=400)
    plant.engage()
    arm = AgentArm(plant, ReachController(ControllerCfg(), __import__("so101_reach.settings", fromlist=["POLICY_DIR"]).POLICY_DIR), SafeTargets(), 0.0)
    cam.warmup(1.0)
    k = cam.intrinsics
    start = arm.open_percent(bus.sync_read("Present_Position")["gripper"])
    for pct in (8, 14, 20, 26, 32, start):
      arm.gripper(pct, "gentle")
      for _ in range(15):
        plant.wait()
      q, _ = arm.q6()
      fr = cam.read()
      tips = jaw_tip_pixels(fr.color)
      kin.fk(q)
      Rg = kin.d.xmat[gr].reshape(3, 3)
      img = fr.color.copy()
      if tips is None:
        print(f"opening {pct:.0f}%: jaw tips not found", flush=True)
        continue
      for side, px in tips.items():
        p_world = np.mean([kin.d.geom_xpos[g] for g in tip_ids[side]], axis=0)
        p_g = Rg.T @ (p_world - kin.d.xpos[gr])
        obs.append((p_g, np.array(px)))
        cv2.circle(img, tuple(int(v) for v in px), 6, (0, 255, 0), 2)
      cv2.imwrite(str(out / f"open_{pct:.0f}.png"), img)
      print(f"opening {pct:.0f}%: tips {tips}", flush=True)
  if len(obs) < 4:
    raise SystemExit("too few jaw-tip observations")
  P = np.array([o[0] for o in obs]); U = np.array([o[1] for o in obs])

  def project(x):
    R = R0 @ Rotation.from_rotvec(x[:3]).as_matrix()
    t = t0 + x[3:]
    c = (P - t) @ R  # gripper -> camera
    return np.column_stack((k.fx * c[:, 0] / c[:, 2] + k.cx, k.fy * c[:, 1] / c[:, 2] + k.cy))

  def res(x):
    # 3 px pixel noise; prior 3 deg / 10 mm on the correction.
    return np.concatenate([((project(x) - U) / 3.0).ravel(), x[:3] / np.radians(3), x[3:] / 0.010])

  before = np.linalg.norm(project(np.zeros(6)) - U, axis=1)
  sol = least_squares(res, np.zeros(6))
  after = np.linalg.norm(project(sol.x) - U, axis=1)
  R = R0 @ Rotation.from_rotvec(sol.x[:3]).as_matrix()
  t = t0 + sol.x[3:]
  report = {"observations": len(obs), "reproj_px_before": before.round(1).tolist(), "reproj_px_after": after.round(1).tolist(),
            "rotation_correction_deg": float(np.degrees(np.linalg.norm(sol.x[:3]))), "translation_correction_mm": (1000 * sol.x[3:]).round(1).tolist()}
  print(json.dumps(report), flush=True)
  (ROOT / "configs" / "wrist_camera_refined.json").write_text(json.dumps({
    "R_gripper_cam": R.tolist(), "t_gripper_cam": t.tolist(), "source": "scripts/calibrate_wrist_jaws.py " + datetime.now().isoformat(timespec="seconds"),
    "based_on": "configs/wrist_camera_extrinsics.json", **report}, indent=2) + "\n")
  (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
  main()
