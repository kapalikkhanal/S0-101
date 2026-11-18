"""Evaluate the Kalman object tracker against raw noisy perception in simulation.

Every seed is one randomized scene from `gen_grasp_data.random_scene` (use seeds disjoint from
the IL training data). Each condition rebuilds the same scene and runs the same scripted probe:

  measure 5 frames parked at home -> open + rise -> measure -> RL travel above the object -> measure ->
  four 2 cm moves at hover height (the arm often hides the object from the head camera here),
  measuring after each -> re-centre above the latest estimate -> scripted grasp on that
  estimate -> 5 cm lift check

Conditions (all noisy ones use the same noise seed per scene):
  truth        exact SimPerception (upper bound)
  noisy        NoisyPerception: Gaussian noise, ~3% outliers, arm-occlusion dropout
  noisy+hold3  noisy + the real pipeline's 3-observation zero-order hold (RealPerception.memory)
  noisy+kf     noisy + TrackedPerception (Kalman filter, gating, occlusion bridging)

Metrics vs world.object_truth(): centre error (xy / z, RMSE and p95), yaw error (bars:
180-deg period, boxes: 90-deg; round shapes skipped), availability of an estimate overall and
while occluded, outliers that reached the output, grasp success, and for the KF the share of
measurements that were confident (= camera parks the real pipeline could skip).
Prints every failed grasp, not only aggregates. JSON report -> results/tracker_eval_<time>.json.

  python scripts/eval_tracker.py --episodes 20 --seed 200000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gen_grasp_data as G  # noqa: E402  (sets MUJOCO_GL before mujoco is imported)
import numpy as np  # noqa: E402

from so101_reach.agent.arm import AgentArm  # noqa: E402
from so101_reach.agent.local_skills import jaw_for, scripted_grasp  # noqa: E402
from so101_reach.agent.perception import Detection, SimPerception  # noqa: E402
from so101_reach.agent.sim_noise import NoiseCfg, NoisyPerception, occluded  # noqa: E402
from so101_reach.agent.sim_world import SimWorld  # noqa: E402
from so101_reach.agent.tracker import KFCfg, TrackedPerception  # noqa: E402
from so101_reach.real.controller import ControllerCfg, ReachController  # noqa: E402
from so101_reach.real.mouse_real import SafeTargets  # noqa: E402
from so101_reach.real.robot import SimRealPlant  # noqa: E402
from so101_reach.settings import AGENT_POLICY_DIR as POLICY  # noqa: E402
from so101_reach.sim import so101  # noqa: E402

CONDITIONS = ("truth", "noisy", "noisy+hold3", "noisy+kf")
WIGGLE_CM = [(2, 0), (-4, 0), (2, 2), (0, -4)]
HOME_FRAMES = 5


class HoldPerception:
  """The real pipeline's fallback: re-report the last measurement for up to `frames` observations."""

  def __init__(self, per, frames: int = 3):
    self.per, self.frames, self.frame = per, frames, 0
    self.table_z = per.table_z
    self.memory: dict[str, tuple[int, list[Detection]]] = {}

  def views(self):
    self.frame += 1
    return self.per.views()

  def find(self, name: str) -> list[Detection]:
    dets = self.per.find(name)
    if dets:
      self.memory[name] = (self.frame, dets)
      return dets
    if name in self.memory and self.frame - self.memory[name][0] <= self.frames:
      return self.memory[name][1]  # the real perception marks these with a note; the pipeline still acts on them
    return []


def wrap(a: float, period: float) -> float:
  return float((a + period / 2) % period - period / 2)


def run(cond: str, seed: int, ctrl, noise: NoiseCfg, kf: KFCfg) -> dict:
  cfg, label, meta = G.random_scene(seed)
  world = SimWorld(cfg)
  world.set_arm({**so101.HOME_JOINT_POS, **so101.FIXED_JOINT_TARGETS, "wrist_roll": world.jmap.to_rad("wrist_roll", 2048)})
  plant = SimRealPlant(world, world.jmap, world.fcal)
  plant.engage()
  for _ in range(25):
    plant.wait()
  arm = AgentArm(plant, ctrl, SafeTargets(), G.TZ)
  base = SimPerception(world)
  base.views = lambda: {}  # measurements only: no rendering
  per = base
  if cond != "truth":
    per = NoisyPerception(base, arm_points=arm.arm_points, cfg=noise, seed=seed * 7 + 1, render=False)
  if cond == "noisy+hold3":
    per = HoldPerception(per, 3)
  elif cond == "noisy+kf":
    per = TrackedPerception(per, [label], cfg=kf, arm_points=arm.arm_points)
  cam = world.camera_model("head")[1]
  period = {"bar": 180.0, "box": 90.0}.get(meta["shape"])
  recs: list[dict] = []
  out = {"seed": seed, "label": label, **meta, "cond": cond}

  def measure(tag: str) -> Detection | None:
    conf = bool(per.confident([label])) if hasattr(per, "confident") else False
    per.views()
    dets = [d for d in per.find(label) if not d.extra.get("note")]
    g = world.object_truth(label)
    truth = np.asarray(g["center"], float)
    r = {"tag": tag, "t": round(float(world.data.time), 2), "confident": conf,
         "occluded": occluded(cam, truth, arm.arm_points(), noise.occl_radius), "available": bool(dets)}
    if dets:
      e = dets[0]
      r["err_xy_mm"] = round(1000 * float(np.linalg.norm(e.center[:2] - truth[:2])), 2)
      r["err_z_mm"] = round(1000 * abs(float(e.center[2] - truth[2])), 2)
      if period:
        r["err_yaw_deg"] = round(abs(wrap(e.yaw_deg - g["yaw_deg"], period)), 2)
    recs.append(r)
    return dets[0] if dets else None

  try:
    for i in range(HOME_FRAMES):  # parked at home: the camera streams several frames
      if i:
        arm.wait(0.2)
      est = measure(f"home{i + 1}") or (est if i else None)
    if est is None:
      return {**out, "aborted": True, "success": False, "note": "not detected at home", "recs": recs}
    width = float(min(est.size[:2]))
    arm.gripper(min(80.0, arm.percent_for_gap(width + 0.02)), "medium")
    G._rise_to(arm, max(est.top_z + 0.03, G.TZ + arm.state()["grip_point_cm"][2] / 100))
    est = measure("risen") or est

    def above(e: Detection) -> str:
      j = jaw_for(arm, e, float(min(e.size[:2])))
      h = G._cm(e.top_z + G.HOVER - G.TZ)
      r = arm.move_to(G._cm(e.center[0]), G._cm(e.center[1]), h, 0.0, j)
      return r if r.startswith("ok") else arm.move_to(G._cm(e.center[0]), G._cm(e.center[1]), h, None, j)

    if not above(est).startswith("ok"):
      return {**out, "aborted": True, "success": False, "note": "travel failed", "recs": recs}
    est = measure("hover") or est
    for i, (dx, dy) in enumerate(WIGGLE_CM):
      arm.move_by(dx, dy, 0.0)
      est = measure(f"wiggle{i + 1}") or est
    above(est)
    ok, note = scripted_grasp(arm, est, G.TZ, None)
    ok = ok and G._lift_verified(arm)
    g = world.object_truth(label)
    out.update({"aborted": False, "success": bool(ok), "note": note[:140],
                "grasp_err_xy_mm": round(1000 * float(np.linalg.norm(est.center[:2] - np.asarray(g["center"])[:2])), 1)})
  finally:
    world.close_renderers()
  out["recs"] = recs
  if hasattr(per, "stats"):
    out["kf_stats"] = dict(per.stats)
  return out


def summarize(eps: list[dict]) -> dict:
  recs = [r for e in eps for r in e["recs"]]
  av = [r for r in recs if r["available"]]

  def stat(key):
    v = np.array([r[key] for r in av if key in r], float)
    if not len(v):
      return None
    return {"rmse": round(float(np.sqrt((v ** 2).mean())), 2), "p95": round(float(np.percentile(v, 95)), 2),
            "max": round(float(v.max()), 2)}

  occ = [r for r in recs if r["occluded"]]
  valid = [e for e in eps if not e["aborted"]]
  return {"episodes": len(eps), "aborted": len(eps) - len(valid),
          "grasp": {"ok": sum(e["success"] for e in valid), "n": len(valid)},
          "measurements": len(recs), "occluded_measurements": len(occ),
          "available": round(len(av) / max(1, len(recs)), 3),
          "available_when_occluded": round(sum(r["available"] for r in occ) / max(1, len(occ)), 3) if occ else None,
          "err_xy_mm": stat("err_xy_mm"), "err_z_mm": stat("err_z_mm"), "err_yaw_deg": stat("err_yaw_deg"),
          "gross_errors_gt30mm": sum(r.get("err_xy_mm", 0) > 30 for r in av),
          "confident_share": round(sum(r["confident"] for r in recs) / max(1, len(recs)), 3)}


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--episodes", type=int, default=20)
  p.add_argument("--seed", type=int, default=200_000, help="first scene seed (keep disjoint from IL training seeds)")
  p.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
  p.add_argument("--sigma-xy-mm", type=float, default=5.0)
  p.add_argument("--p-outlier", type=float, default=0.03)
  p.add_argument("--p-miss", type=float, default=0.02)
  p.add_argument("--speed", type=float, default=0.5)
  p.add_argument("--out", type=Path, default=None)
  args = p.parse_args()

  noise = NoiseCfg(sigma_xy=args.sigma_xy_mm / 1000, p_outlier=args.p_outlier, p_miss=args.p_miss)
  kf = KFCfg(meas_xy=noise.sigma_xy, meas_z=noise.sigma_z)
  ctrl = ReachController(ControllerCfg(speed_factor=args.speed), policy_dir=POLICY)
  results: dict[str, list[dict]] = {c: [] for c in args.conditions}
  t0 = time.time()
  for i in range(args.episodes):
    seed = args.seed + i
    line = []
    for c in args.conditions:
      r = run(c, seed, ctrl, noise, kf)
      results[c].append(r)
      line.append(f"{c}={'--' if r['aborted'] else ('OK' if r['success'] else 'FAIL')}")
    print(f"[eval] {i + 1}/{args.episodes} seed {seed} {results[args.conditions[0]][-1]['label']:16s} " + " ".join(line), flush=True)

  summary = {c: summarize(rs) for c, rs in results.items()}
  print(f"\n[eval] {args.episodes} scenes (seeds {args.seed}-{args.seed + args.episodes - 1}), {time.time() - t0:.0f} s; "
        f"noise sigma_xy {args.sigma_xy_mm} mm, outliers {args.p_outlier:.0%}, misses {args.p_miss:.0%}")
  cell = lambda d, k="rmse": "-" if d is None else f"{d[k]:.1f}"  # noqa: E731
  head = (f"{'condition':12s} {'grasp':>7s} {'xy rmse':>8s} {'xy p95':>7s} {'z rmse':>7s} {'yaw rmse':>8s} "
          f"{'avail':>6s} {'avail|occl':>10s} {'>30mm':>6s} {'confident':>9s}")
  print(head)
  print("-" * len(head))
  for c, s in summary.items():
    occ = "-" if s["available_when_occluded"] is None else f"{s['available_when_occluded']:.2f}"
    print(f"{c:12s} {s['grasp']['ok']:>3d}/{s['grasp']['n']:<3d} {cell(s['err_xy_mm']):>8s} {cell(s['err_xy_mm'], 'p95'):>7s} "
          f"{cell(s['err_z_mm']):>7s} {cell(s['err_yaw_deg']):>8s} {s['available']:>6.2f} {occ:>10s} "
          f"{s['gross_errors_gt30mm']:>6d} {s['confident_share']:>9.2f}")
  print("  (errors in mm / deg over measurements that returned an estimate; avail|occl = estimate available while the arm "
        "blocked the head camera; confident = measurements where the real pipeline could skip parking the arm)")
  print("\n[eval] failed grasps per condition:")
  for c, rs in results.items():
    fails = [r for r in rs if not r["success"]]
    print(f"  {c}: {len(fails)}")
    for r in fails:
      extra = "" if r["aborted"] else f" (estimate off by {r.get('grasp_err_xy_mm', '?')} mm)"
      print(f"    seed {r['seed']} {r['label']:16s} {'ABORT ' if r['aborted'] else ''}{r['note'][:100]}{extra}")

  out = args.out or so101.PROJECT_ROOT / "results" / f"tracker_eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "noise": vars(noise),
                             "kf": vars(kf), "summary": summary, "episodes": results}, indent=1, default=str))
  print(f"\n[eval] report -> {out}")


if __name__ == "__main__":
  main()
