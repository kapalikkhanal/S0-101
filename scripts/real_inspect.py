"""READ-ONLY inspection of the real SO-101 follower: pings the servos and prints their
state. Never writes to the bus (the arm does not move, torque is not changed).

  python scripts/real_inspect.py --port /dev/ttyACM0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from so101_reach.real.feetech_bus import DEFAULT_CALIB, JOINT_ORDER, FeetechBus, load_calibration
from so101_reach.real.joint_map import FOLLOWER_PORT

REGS = (
  "Torque_Enable", "Present_Position", "Goal_Position", "Present_Velocity", "Present_Load",
  "Homing_Offset", "Min_Position_Limit", "Max_Position_Limit", "P_Coefficient", "D_Coefficient",
  "I_Coefficient", "Operating_Mode", "Acceleration", "Torque_Limit", "Max_Torque_Limit", "Lock",
  "Present_Voltage", "Present_Temperature",
)


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--port", default=FOLLOWER_PORT)
  p.add_argument("--calibration", type=Path, default=DEFAULT_CALIB)
  args = p.parse_args()
  calib = load_calibration(args.calibration)
  with FeetechBus(args.port, calib=calib) as bus:
    print(f"port {args.port}, calibration {args.calibration}")
    for name in JOINT_ORDER:
      print(f"  {name:13s} id {calib[name].id}: model {bus.ping(calib[name].id)}")
    rows = {reg: bus.sync_read(reg) for reg in REGS}
  print(f"\n{'register':20s}" + "".join(f"{n:>14s}" for n in JOINT_ORDER))
  for reg, vals in rows.items():
    print(f"{reg:20s}" + "".join(f"{vals[n]:>14d}" for n in JOINT_ORDER))
  print(f"\n{'calibration':20s}" + "".join(f"{n:>14s}" for n in JOINT_ORDER))
  for field in ("homing_offset", "range_min", "range_max", "drive_mode"):
    print(f"{field:20s}" + "".join(f"{getattr(calib[n], field):>14d}" for n in JOINT_ORDER))


if __name__ == "__main__":
  main()
