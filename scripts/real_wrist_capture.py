"""Read-only wrist (D405) capture: colour, depth (.npy), joint angles and intrinsics.

  .venv/bin/python scripts/real_wrist_capture.py --out results/hardware_2026-11-07/wrist_hover
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from so101_reach.real.cameras import open_camera
from so101_reach.real.feetech_bus import JOINT_ORDER, FeetechBus, load_calibration
from so101_reach.real.joint_map import FOLLOWER_PORT, JointMap


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--out", type=Path, required=True)
  args = p.parse_args()
  args.out.mkdir(parents=True, exist_ok=True)
  jmap, fcal = JointMap.load(), load_calibration()
  with FeetechBus(FOLLOWER_PORT, calib=fcal) as bus:
    raw = bus.sync_read("Present_Position")
  q = [jmap.to_rad(n, raw[n]) for n in JOINT_ORDER]
  with open_camera("wrist") as cam:
    cam.warmup(1.0)
    frames = [cam.read() for _ in range(5)]
    k = cam.intrinsics
  depth = np.median(np.stack([f.depth for f in frames]), axis=0).astype(np.float32)  # temporal median
  cv2.imwrite(str(args.out / "color.png"), frames[-1].color)
  np.save(args.out / "depth.npy", depth)
  vis = cv2.applyColorMap(cv2.convertScaleAbs(np.clip(depth, 0, 0.3), alpha=255 / 0.3), cv2.COLORMAP_JET)
  vis[depth <= 0] = 0
  cv2.imwrite(str(args.out / "depth_vis.png"), vis)
  (args.out / "meta.json").write_text(json.dumps({"q": q, "ticks": raw, "intrinsics": k.__dict__}, indent=2, default=str) + "\n")
  print("valid depth", round(float((depth > 0).mean()), 2), "saved", args.out)


if __name__ == "__main__":
  main()
