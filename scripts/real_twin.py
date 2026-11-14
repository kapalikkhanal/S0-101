"""Step 1 - READ-ONLY digital twin + joint check for the real SO-101 follower.

Torque is never enabled and nothing is written to the servos: move the arm by hand and the
MuJoCo window mirrors it (joint angles, TCP position in the robot base frame).

Checks:
  1. Direction: move one joint at a time; the twin must move the same way.
     If a joint is mirrored, flip its "sign" in configs/real_joint_map.json.
  2. Tap test: touch the closed gripper tip on the table at 5-10 spots spread over the
     workspace and press Enter in this terminal at each one. The tip height should be
     ~0-5 mm with a spread of a few mm; a large spread means a joint offset/sign is wrong.
     Type 'q' + Enter to finish and print the summary.

  python scripts/real_twin.py               # window + terminal
  python scripts/real_twin.py --no-viewer   # terminal only
"""

from __future__ import annotations

import argparse
import os
import threading
import time

os.environ.setdefault("MUJOCO_GL", "glfw")

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402

from so101_reach.real.feetech_bus import DEFAULT_CALIB, FeetechBus, load_calibration  # noqa: E402
from so101_reach.real.joint_map import DEFAULT_MAP_PATH, FOLLOWER_PORT, JointMap  # noqa: E402
from so101_reach.sim import so101  # noqa: E402


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--port", default=FOLLOWER_PORT)
  p.add_argument("--no-viewer", action="store_true")
  p.add_argument("--duration", type=float, default=0.0, help="stop after N seconds (0 = until 'q')")
  args = p.parse_args()

  jmap = JointMap.load()
  if not DEFAULT_MAP_PATH.exists():
    jmap.save()
    print(f"[twin] wrote default joint map {DEFAULT_MAP_PATH}")
  model = so101.make_cpu_model()
  data = mujoco.MjData(model)
  qadr = {n: model.jnt_qposadr[model.joint(n).id] for n in so101.ALL_JOINTS}
  site = model.site(so101.TCP_SITE).id
  taps: list[np.ndarray] = []
  stop = threading.Event()

  def stdin_loop() -> None:
    print("[twin] Enter = record tap point, q + Enter = finish")
    while not stop.is_set():
      try:
        line = input().strip().lower()
      except EOFError:
        return
      if line == "q":
        stop.set()
      else:
        taps.append(data.site_xpos[site].copy())
        t = taps[-1]
        print(f"[tap {len(taps)}] TCP x={t[0]:+.4f} y={t[1]:+.4f} z={t[2]:+.4f} m")

  viewer = None if args.no_viewer else mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
  threading.Thread(target=stdin_loop, daemon=True).start()
  t0 = time.perf_counter()
  last_print = 0.0
  with FeetechBus(args.port, calib=load_calibration(DEFAULT_CALIB)) as bus:
    torque = bus.sync_read("Torque_Enable")
    if any(torque.values()):
      print(f"[twin] WARNING: torque is enabled on {[k for k, v in torque.items() if v]} (this script will not change it)")
    while not stop.is_set():
      ticks = bus.sync_read("Present_Position")
      q = {n: jmap.to_rad(n, ticks[n]) for n in so101.ALL_JOINTS}
      for n, v in q.items():
        data.qpos[qadr[n]] = v
      mujoco.mj_forward(model, data)
      tcp = data.site_xpos[site]
      now = time.perf_counter() - t0
      if viewer is not None:
        if not viewer.is_running():
          break
        viewer.set_texts(
          (
            mujoco.mjtFontScale.mjFONTSCALE_150.value,
            mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
            "TCP [m]\n" + "\n".join(so101.ARM_JOINTS),
            f"{tcp[0]:+.3f} {tcp[1]:+.3f} {tcp[2]:+.3f}\n" + "\n".join(f"{np.degrees(q[n]):+7.1f} deg  ({ticks[n]})" for n in so101.ARM_JOINTS),
          )
        )
        viewer.sync()
      if now - last_print > 0.5:
        last_print = now
        js = " ".join(f"{n[:5]} {np.degrees(q[n]):+6.1f}" for n in so101.ARM_JOINTS)
        print(f"\r[twin] {js} | TCP {tcp[0]:+.3f} {tcp[1]:+.3f} {tcp[2]:+.3f} m | contacts {data.ncon}   ", end="", flush=True)
      if args.duration and now > args.duration:
        break
      time.sleep(0.02)
  stop.set()
  if viewer is not None:
    viewer.close()
  print()
  if taps:
    z = np.array([t[2] for t in taps]) * 1e3
    print(f"[tap test] {len(taps)} points: tip height mean {z.mean():+.1f} mm, std {z.std():.1f} mm, range [{z.min():+.1f}, {z.max():+.1f}] mm")
    print("           expected: mean 0..+5 mm, std <= ~3 mm. Larger -> check signs/zero ticks in", DEFAULT_MAP_PATH)


if __name__ == "__main__":
  main()
