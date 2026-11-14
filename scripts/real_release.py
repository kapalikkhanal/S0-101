"""Safely let the real arm go limp: hold the present pose, count down while you support the
arm by hand, then disable torque on all servos.

  python scripts/real_release.py            # 5 s countdown
  python scripts/real_release.py --now      # no countdown
  python scripts/real_release.py --leader   # the leader arm instead of the follower
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from so101_reach.real.feetech_bus import JOINT_ORDER, FeetechBus, load_calibration
from so101_reach.real.joint_map import FOLLOWER_PORT
from so101_reach.real.teleop import LEADER_CALIB, LEADER_PORT


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--port", default=FOLLOWER_PORT)
  p.add_argument("--countdown", type=float, default=5.0)
  p.add_argument("--now", action="store_true")
  p.add_argument("--leader", action="store_true", help="release the leader arm")
  args = p.parse_args()
  calib = load_calibration(LEADER_CALIB) if args.leader else load_calibration()
  if args.leader:
    args.port = LEADER_PORT
  with FeetechBus(args.port, calib=calib) as bus:
    pos = bus.sync_read("Present_Position")
    bus.sync_write("Goal_Position", {n: int(np.clip(pos[n], calib[n].range_min, calib[n].range_max)) for n in JOINT_ORDER})
    if not args.now:
      print("Holding pose. SUPPORT THE ARM BY HAND - torque goes off in:")
      for k in range(int(args.countdown), 0, -1):
        print(f"  {k}...")
        time.sleep(1.0)
    bus.sync_write("Torque_Enable", {n: 0 for n in JOINT_ORDER})
    print("torque OFF:", bus.sync_read("Torque_Enable"))


if __name__ == "__main__":
  main()
