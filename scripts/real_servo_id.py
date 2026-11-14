"""Small-step servo identification on the real follower (MOVES THE ARM slightly).

For each arm joint (one at a time): +step, hold, -step back, recording Present_Position at
~200 Hz. Reports rise time, overshoot and steady-state error (tracking deadband) for each
Torque_Limit setting. The arm must already be in a safe pose; other joints hold position.

  python scripts/real_servo_id.py --step-deg 4 --torque-limits 400 1000
  python scripts/real_servo_id.py --record      # full traces (steps 2/5/8 deg, ramps 10/30 deg/s) -> logs/real/servo_id_*.npz
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from so101_reach.real.feetech_bus import JOINT_ORDER, FeetechBus, load_calibration
from so101_reach.real.joint_map import FOLLOWER_PORT
from so101_reach.sim import so101

TICK_DEG = 360 / 4096


def record(bus: FeetechBus, name: str, seconds: float) -> tuple[np.ndarray, np.ndarray]:
  t0 = time.perf_counter()
  ts, ps = [], []
  while time.perf_counter() - t0 < seconds:
    ps.append(bus.read(name, "Present_Position"))
    ts.append(time.perf_counter() - t0)
  return np.array(ts), np.array(ps)


def record_traces(bus: FeetechBus, calib, out_path) -> None:
  """Steps of 2/5/8 deg both ways and ramps at 10/30 deg/s per arm joint; all joints'
  positions are logged (sync read) at ~400 Hz so coupling through gravity is captured."""
  traces = {}
  pose = bus.sync_read("Present_Position")
  for n in so101.ARM_JOINTS:
    c = calib[n]
    for kind, amp in (("step", 2), ("step", 5), ("step", 8), ("ramp", 10), ("ramp", 30)):
      for direction in (1, -1):
        start = bus.read(n, "Present_Position")
        sgn = direction if c.range_min + 120 < start + direction * 100 < c.range_max - 120 else -direction
        goal_ticks = int(round(amp / TICK_DEG)) * sgn if kind == "step" else int(round(8 / TICK_DEG)) * sgn
        t0 = time.perf_counter()
        ts, cmd, pos = [], [], []
        dur = 1.0 if kind == "step" else 8.0 / amp + 0.8
        while (t := time.perf_counter() - t0) < dur:
          if kind == "step":
            g = start + goal_ticks
          else:
            g = start + int(round(np.clip(t * amp / 8.0, 0, 1) * goal_ticks))
          bus.write(n, "Goal_Position", int(g))
          ps = bus.sync_read("Present_Position")
          ts.append(t)
          cmd.append(g)
          pos.append([ps[j] for j in so101.ARM_JOINTS])
        traces[f"{n}/{kind}{amp}/{'+' if sgn > 0 else '-'}"] = (np.array(ts), np.array(cmd), np.array(pos))
        bus.write(n, "Goal_Position", int(start))
        time.sleep(0.6)
  np.savez(out_path, pose=np.array([pose[j] for j in so101.ARM_JOINTS]), **{f"{k}|t": v[0] for k, v in traces.items()},
           **{f"{k}|cmd": v[1] for k, v in traces.items()}, **{f"{k}|pos": v[2] for k, v in traces.items()})
  print(f"saved {len(traces)} traces -> {out_path}")


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--port", default=FOLLOWER_PORT)
  p.add_argument("--step-deg", type=float, default=4.0)
  p.add_argument("--torque-limits", type=int, nargs="+", default=[400, 1000])
  p.add_argument("--hold", type=float, default=1.2)
  p.add_argument("--record", action="store_true", help="record full traces for model fitting")
  args = p.parse_args()
  calib = load_calibration()
  step = int(round(args.step_deg / TICK_DEG))
  with FeetechBus(args.port, calib=calib) as bus:
    pos = bus.sync_read("Present_Position")
    bus.sync_write("Goal_Position", {n: int(np.clip(pos[n], calib[n].range_min, calib[n].range_max)) for n in JOINT_ORDER})
    bus.sync_write("Torque_Enable", {n: 1 for n in JOINT_ORDER})
    if args.record:
      from datetime import datetime

      out = so101.PROJECT_ROOT / "logs" / "real" / f"servo_id_{datetime.now():%Y%m%d_%H%M%S}.npz"
      out.parent.mkdir(parents=True, exist_ok=True)
      for n in so101.ARM_JOINTS:
        bus.write(n, "Torque_Limit", 1000)
      record_traces(bus, calib, out)
      args.torque_limits = []
    print(f"step {step} ticks ({step * TICK_DEG:.1f} deg); rise = time to 90 %, ss = steady-state error after {args.hold:.1f} s")
    for lim in args.torque_limits:
      for n in so101.ARM_JOINTS:
        bus.write(n, "Torque_Limit", lim)
      for n in so101.ARM_JOINTS:
        c = calib[n]
        start = bus.read(n, "Present_Position")
        sgn = 1 if start + step <= c.range_max - 20 else -1
        res = []
        for goal in (start + sgn * step, start):
          before = bus.read(n, "Present_Position")
          bus.write(n, "Goal_Position", int(goal))
          t, x = record(bus, n, args.hold)
          d = goal - before
          prog = (x - before) / d if d else np.ones_like(x)
          rise = t[np.argmax(prog >= 0.9)] if np.any(prog >= 0.9) else float("nan")
          over = max(0.0, (prog.max() - 1.0)) * abs(d) * TICK_DEG
          ss = (goal - x[-int(len(x) * 0.2):].mean()) * TICK_DEG
          res.append(f"rise {rise:.3f}s over {over:.2f}deg ss {ss:+.2f}deg")
        print(f"  torque {lim:4d} {n:13s} out: {res[0]} | back: {res[1]}")
    for n in so101.ARM_JOINTS:
      bus.write(n, "Torque_Limit", 400)
    pos = bus.sync_read("Present_Position")
    bus.sync_write("Goal_Position", {n: int(np.clip(pos[n], calib[n].range_min, calib[n].range_max)) for n in JOINT_ORDER})
    print("done; holding present pose (torque on, limit 400)")


if __name__ == "__main__":
  main()
