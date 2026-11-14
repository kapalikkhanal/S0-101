"""Real-robot reaching runner.

Modes:
  sim  - deployment controller (ONNX policy + real-style sensing) on a CPU MuJoCo plant.
         No hardware. Validates the exact code that will drive the arm.
  dry  - Step 2. Reads the real arm (torque OFF, nothing is written). For each target you
         type, the motion is predicted in simulation from the arm's *measured* pose,
         replayed in the twin window, and the servo command stream it would send is
         checked against the calibrated servo limits. Saved to logs/real/.
  live - Step 3. MOVES THE ARM. Engages torque holding the present pose (no jump), reaches
         the target at reduced speed/torque, then holds. Ctrl+C or a safety stop -> hold.
         Torque stays on at the end (the arm would fall otherwise); --release-at-end to drop.
         --sim-bus runs the identical code against a simulated servo bus.

  python scripts/real_run.py --mode sim
  python scripts/real_run.py --mode dry           # then type: 0.20 0.10 0.15 | r | q
  python scripts/real_run.py --mode live --nudge 0 0 0.03 --speed 0.25 --torque-limit 400
  python scripts/real_run.py --mode live --target 0.22 0.0 0.15 --speed 0.25
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from datetime import datetime

os.environ.setdefault("MUJOCO_GL", "glfw")

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402

from so101_reach.real.controller import POLICY_DIR, ControllerCfg, ReachController, SimPlant, rollout  # noqa: E402
from so101_reach.real.feetech_bus import FeetechBus, load_calibration  # noqa: E402
from so101_reach.real.joint_map import FOLLOWER_PORT, JointMap  # noqa: E402
from so101_reach.real.robot import RealPlant, SafetyStop, SimRealPlant, SimServoBus, run_live  # noqa: E402
from so101_reach.sim import so101  # noqa: E402
from so101_reach.sim.workspace import load_bank  # noqa: E402

LOG_DIR = so101.PROJECT_ROOT / "logs" / "real"
LIMITS_PATH = so101.PROJECT_ROOT / "configs" / "real_workspace.json"


def site_limits() -> dict[str, float]:
  """Keep-out limits of this installation (walls, fixtures) for TCP targets."""
  import json

  return json.loads(LIMITS_PATH.read_text()) if LIMITS_PATH.exists() else {}


class Workspace:
  def __init__(self):
    self.pts = load_bank("eval").tcp_pos

  def check(self, xyz: np.ndarray) -> tuple[np.ndarray, float]:
    d = np.linalg.norm(self.pts - xyz, axis=1)
    i = int(d.argmin())
    return (self.pts[i] if d[i] > 0.01 else xyz), float(d[i])

  def random(self, rng) -> np.ndarray:
    return self.pts[rng.integers(len(self.pts))]


def command_report(traj: dict, jmap: JointMap, calib) -> list[str]:
  lines, ok = [], True
  ticks = np.stack([[jmap.to_ticks(n, v) for n, v in zip(so101.ARM_JOINTS, jt, strict=True)] for jt in traj["joint_target"]])
  step = np.abs(np.diff(ticks, axis=0)).max(axis=0) if len(ticks) > 1 else np.zeros(4)
  for j, n in enumerate(so101.ARM_JOINTS):
    c = calib[n]
    inside = c.range_min <= ticks[:, j].min() and ticks[:, j].max() <= c.range_max
    ok &= inside
    lines.append(
      f"  {n:13s} goal ticks [{ticks[:, j].min():4d}, {ticks[:, j].max():4d}] within calib [{c.range_min}, {c.range_max}]: "
      f"{'yes' if inside else 'NO'}; max change/step {step[j]:3d} ticks ({np.degrees(step[j] * 2 * np.pi / 4096):.1f} deg)"
    )
  lines.append(f"  command stream {'OK' if ok else 'has OUT-OF-RANGE goals'}")
  return lines


def run_sim(args) -> None:
  rng = np.random.default_rng(args.seed)
  ws = Workspace()
  ctrl = ReachController(ControllerCfg(speed_factor=args.speed), policy_dir=args.policy_dir)
  q0 = np.array([so101.HOME_JOINT_POS[n] for n in so101.ARM_JOINTS])
  plant = SimPlant(q0, quantize=True, actuator_model=args.sim_actuator)
  for k in range(args.num_targets):
    if args.target is not None:
      requested = np.array(args.target, dtype=float)
      tgt, distance = ws.check(requested)
      if distance > 0.01:
        print(f"[sim] requested target snapped to sampled workspace (distance {distance:.3f} m)")
    else:
      tgt = ws.random(rng)
    tr = rollout(ctrl, plant, tgt, int(args.horizon / plant.dt))
    e = tr["err"] * 1e3
    hit = np.where(e < 5)[0]
    print(f"[sim] target {k + 1}: {np.round(tgt, 3)} -> final {e[-1]:.2f} mm, <5 mm at {(hit[0] + 1) * plant.dt if len(hit) else float('nan'):.2f} s")


def run_dry(args) -> None:
  rng = np.random.default_rng(args.seed)
  ws = Workspace()
  jmap = JointMap.load()
  calib = load_calibration()
  ctrl = ReachController(ControllerCfg(speed_factor=args.speed), policy_dir=args.policy_dir)
  model = so101.make_cpu_model()
  data = mujoco.MjData(model)
  qadr = {n: model.jnt_qposadr[model.joint(n).id] for n in so101.ALL_JOINTS}
  LOG_DIR.mkdir(parents=True, exist_ok=True)
  viewer = None if args.no_viewer else mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)

  def show(q_arm: np.ndarray, label: str, extra: str = "") -> None:
    for n, v in zip(so101.ARM_JOINTS, q_arm, strict=True):
      data.qpos[qadr[n]] = v
    mujoco.mj_forward(model, data)
    if viewer is not None and viewer.is_running():
      viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150.value, mujoco.mjtGridPos.mjGRID_TOPLEFT.value, label, extra))
      viewer.sync()

  with FeetechBus(args.port, calib=calib) as bus:
    torque = bus.sync_read("Torque_Enable")
    print(f"[dry] connected, torque {torque} (this mode never writes to the servos)")

    def read_q() -> np.ndarray:
      t = bus.sync_read("Present_Position")
      for n in so101.FIXED_JOINTS:
        data.qpos[qadr[n]] = jmap.to_rad(n, t[n])
      return np.array([jmap.to_rad(n, t[n]) for n in so101.ARM_JOINTS])

    print("[dry] type a target 'x y z' (m, robot base frame), 'r' random, 'q' quit")
    while True:
      show(read_q(), "LIVE", "real arm (torque off)")
      try:
        line = input("target> ").strip().lower()
      except EOFError:
        break
      if line in ("q", "quit"):
        break
      if line == "r":
        xyz = ws.random(rng)
      else:
        try:
          xyz = np.array([float(v) for v in line.replace(",", " ").split()][:3])
          assert len(xyz) == 3
        except (ValueError, AssertionError):
          print("  expected 'x y z', 'r' or 'q'")
          continue
      tgt, dist = ws.check(xyz)
      if dist > 0.01:
        print(f"  {np.round(xyz, 3)} is outside the verified workspace -> nearest reachable {np.round(tgt, 3)} ({dist * 1e3:.0f} mm away)")
      q0 = read_q()
      tr = rollout(ctrl, SimPlant(q0, quantize=True, actuator_model=args.sim_actuator), tgt, int(args.horizon / 0.02))
      e = tr["err"] * 1e3
      hit = np.where(e < 10)[0]
      print(f"  start TCP {np.round(ctrl.kin.tcp(q0), 3)} -> target {np.round(tgt, 3)}")
      print(f"  predicted: <10 mm after {(hit[0] + 1) * 0.02 if len(hit) else float('nan'):.2f} s, final error {e[-1]:.2f} mm, "
            f"peak joint speed {np.degrees(np.abs(np.diff(tr['q'], axis=0)).max() / 0.02):.0f} deg/s")
      for ln in command_report(tr, jmap, calib):
        print(ln)
      stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
      np.savez(LOG_DIR / f"dry_{stamp}.npz", target=tgt, q0=q0, **tr)
      if viewer is not None:
        for k, q in enumerate(tr["q"][:: max(1, int(args.playback_skip))]):
          show(q, "PREVIEW (simulated)", f"t = {k * 0.02 * args.playback_skip:.2f} s  err {e[min(k * args.playback_skip, len(e) - 1)]:.1f} mm")
          time.sleep(0.02 * args.playback_skip)
  if viewer is not None:
    viewer.close()


def run_live_mode(args) -> None:
  jmap = JointMap.load()
  calib = load_calibration()
  ws = Workspace()
  ctrl = ReachController(ControllerCfg(speed_factor=args.speed), policy_dir=args.policy_dir)
  if args.sim_bus:
    q_start = np.array([so101.HOME_JOINT_POS[n] for n in so101.ARM_JOINTS])
    bus_ctx = _NullCtx(SimServoBus(jmap, q_start))
    plant_cls = SimRealPlant
  else:
    bus_ctx = FeetechBus(args.port, calib=calib)
    plant_cls = RealPlant
  with bus_ctx as bus:
    plant = plant_cls(bus, jmap, calib, torque_limit=args.torque_limit)
    q0 = plant.read_q()
    tcp0 = ctrl.kin.tcp(q0)
    kin = ctrl.kin
    mujoco.mj_forward(kin.model, kin.data)
    deep = [c for c in kin.data.contact[: kin.data.ncon] if c.dist < -0.005]
    if tcp0[2] < -0.005 or deep:
      raise SystemExit(f"measured pose is implausible (TCP z {tcp0[2]:.3f} m, {len(deep)} penetrating contacts): check configs/real_joint_map.json; refusing")
    if args.sequence:
      raws = [np.array([float(v) for v in t.split(",")]) for t in args.sequence.split(";")]
    else:
      raws = [tcp0 + np.array(args.nudge) if args.nudge is not None else np.array(args.target)]
    targets = []
    for raw in raws:  # Validate every target before any motion.
      tgt, dist = ws.check(raw)
      if dist > 0.01:
        raise SystemExit(f"target {np.round(raw, 3)} is outside the verified workspace ({dist * 1e3:.0f} mm); refusing")
      if tgt[2] < args.min_z:
        raise SystemExit(f"target z {tgt[2]:.3f} m below --min-z {args.min_z}; refusing")
      lim = site_limits()
      for k, ax in enumerate("xyz"):
        lo, hi = lim.get(f"{ax}_min", -np.inf), lim.get(f"{ax}_max", np.inf)
        if not lo <= tgt[k] <= hi:
          raise SystemExit(f"target {ax}={tgt[k]:+.3f} m outside site limits [{lo}, {hi}] ({LIMITS_PATH.name}: {lim.get('note', '')}); refusing")
      targets.append(tgt)
    print(f"[live] start TCP {np.round(tcp0, 4)}; {len(targets)} target(s), speed {args.speed}, "
          f"torque limit {args.torque_limit}/1000{' [SIM BUS]' if args.sim_bus else ''}")
    for k in range(int(args.countdown), 0, -1):
      print(f"[live] moving in {k} s", flush=True)
      time.sleep(1.0)
    plant.engage()
    print("[live] torque ON, holding present pose")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    try:
      for i, tgt in enumerate(targets):
        start = ctrl.kin.tcp(plant.read_q())
        label = f" ({args.labels.split(';')[i]})" if args.labels else ""
        status = "ok"
        tr = None
        try:
          tr = run_live(plant, ctrl, tgt, args.duration)
        except SafetyStop as e:
          status = f"SAFETY STOP: {e}"
          tr = e.record
        plant.hold()
        tcp1 = ctrl.kin.tcp(plant.read_q())
        print(f"[live] {i + 1}/{len(targets)}{label}: {np.round(start, 3)} -> {np.round(tgt, 3)} "
              f"({np.linalg.norm(tgt - start) * 1e3:.0f} mm): {status}, final error {np.linalg.norm(tgt - tcp1) * 1e3:.1f} mm"
              f"{'' if tr is None else f' in {len(tr["err"]) * plant.dt:.2f} s'}", flush=True)
        if tr is not None:
          stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
          np.savez(LOG_DIR / f"live_{'sim_' if args.sim_bus else ''}{stamp}_{i}.npz", target=tgt, q0=q0, status=status, **tr)
        if status != "ok":
          break
        if args.dwell > 0 and i < len(targets) - 1:
          if args.sim_bus:
            for _ in range(int(args.dwell / plant.dt)):
              plant.wait()
          else:
            time.sleep(args.dwell)
    except KeyboardInterrupt:
      print("[live] interrupted (Ctrl+C)")
    finally:
      plant.hold()
      if args.release_at_end:
        plant.release()
    print(f"[live] done, loop overruns {plant.overruns}; torque "
          + ("released" if args.release_at_end else "still ON (arm holds pose); scripts/real_release.py to let it go limp"))


class _NullCtx:
  def __init__(self, obj):
    self.obj = obj

  def __enter__(self):
    return self.obj

  def __exit__(self, *exc):
    return False


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--mode", choices=["sim", "dry", "live"], required=True)
  p.add_argument("--target", type=float, nargs=3, default=None, help="live: absolute target x y z [m]")
  p.add_argument("--nudge", type=float, nargs=3, default=None, help="live: target = current TCP + dx dy dz [m]")
  p.add_argument("--sequence", type=str, default=None, help='live: "x,y,z;x,y,z;..." visited in order')
  p.add_argument("--labels", type=str, default=None, help='live: ";"-separated names printed per target')
  p.add_argument("--dwell", type=float, default=2.0, help="live: seconds to hold at each target")
  p.add_argument("--countdown", type=float, default=0.0, help="live: seconds to wait before engaging")
  p.add_argument("--torque-limit", type=int, default=400, help="live: servo Torque_Limit (0-1000)")
  p.add_argument("--duration", type=float, default=6.0, help="live: max seconds")
  p.add_argument("--min-z", type=float, default=0.08, help="live: refuse targets lower than this [m]")
  p.add_argument("--release-at-end", action="store_true", help="live: disable torque at the end (arm falls)")
  p.add_argument("--sim-bus", action="store_true", help="live: run against a simulated servo bus")
  p.add_argument("--sim-actuator", choices=["menagerie", "identified"], default="menagerie",
                 help="servo model used for sim/dry predictions (identified = fitted on this arm)")
  p.add_argument("--policy-dir", type=Path, default=POLICY_DIR, help="exported policy (policy.onnx + contract)")
  p.add_argument("--port", default=FOLLOWER_PORT)
  p.add_argument("--speed", type=float, default=1.0, help="scale of the per-step joint increment limit")
  p.add_argument("--horizon", type=float, default=4.0)
  p.add_argument("--num-targets", type=int, default=5)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--no-viewer", action="store_true")
  p.add_argument("--playback-skip", type=int, default=1)
  args = p.parse_args()
  if args.mode in ("dry", "live") and not args.sim_bus:
    from so101_reach.settings import require_hardware_config
    require_hardware_config()
  if args.mode == "live" and sum(x is not None for x in (args.target, args.nudge, args.sequence)) != 1:
    p.error("live mode needs exactly one of --target, --nudge or --sequence")
  {"sim": run_sim, "dry": run_dry, "live": run_live_mode}[args.mode](args)


if __name__ == "__main__":
  main()
