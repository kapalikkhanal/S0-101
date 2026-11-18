"""Test 4 - reach accuracy under the reality gap: RL travel vs IK travel on the agent's arm stack.

Same AgentArm the manipulation pipeline uses (identified servo model, real torque limits), from
home to random grip-point targets and back. `travel_controller="rl"` uses the exported PPO policy
for the long move; "ik" uses the checked straight-line IK plan. The error is measured on GROUND
TRUTH joint angles, so encoder noise / calibration bias (levels 1, 2) show up as real error the
controller cannot see.

Feeds graphs: sim vs noisy reach error per controller, workspace error map, reach time.

  python scripts/report/t04_reach_sim2real.py --n 150
Results: results/report/reach_sim2real.jsonl
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from common import RESULTS, TZ, append, controller, done_keys
from so101_reach.agent.arm import AgentArm
from so101_reach.agent.sim_reality import make_world
from so101_reach.agent.sim_world import WorldCfg
from so101_reach.real.feetech_bus import JOINT_ORDER
from so101_reach.real.mouse_real import SafeTargets
from so101_reach.real.robot import SimRealPlant
from so101_reach.sim import so101


def targets(n: int, seed: int = 31) -> np.ndarray:
  rng = np.random.default_rng(seed)
  return np.stack([rng.uniform(0.14, 0.32, n), rng.uniform(-0.20, 0.16, n), rng.uniform(0.05, 0.20, n)], 1)


def run(ctrl_name: str, level: float, tgts: np.ndarray, path, seed: int = 5) -> None:
  done = done_keys(path, ("controller", "level", "i"))
  todo = [i for i in range(len(tgts)) if (ctrl_name, level, i) not in done]
  if not todo:
    return
  world = make_world(WorldCfg(TZ), seed, level)
  world.set_arm({**so101.HOME_JOINT_POS, **so101.FIXED_JOINT_TARGETS, "wrist_roll": world.jmap.to_rad("wrist_roll", 2048)})
  plant = SimRealPlant(world, world.jmap, world.fcal)
  plant.engage()
  for _ in range(25):
    plant.wait()
  arm = AgentArm(plant, controller(0.5), SafeTargets(), TZ)
  arm.travel_controller = ctrl_name
  true_q6 = lambda: np.array([world.data.qpos[world.qadr[n]] for n in JOINT_ORDER])  # noqa: E731
  for i in todo:
    x, y, h = tgts[i]
    t0, w0 = float(world.data.time), time.time()
    r = arm.move_to(100 * x, 100 * y, 100 * h, None, None)
    for _ in range(10):
      plant.wait()
    goal = np.array([x, y, TZ + h])
    gp_true = arm.geom(true_q6())["gp"]
    gp_seen = arm.geom(arm.q6()[0])["gp"]
    row = {"controller": ctrl_name, "level": level, "i": i, "target": [round(float(v), 4) for v in goal],
           "status": r[:80], "ok": r.startswith("ok"),
           "err_true_mm": round(1000 * float(np.linalg.norm(gp_true - goal)), 2),
           "err_seen_mm": round(1000 * float(np.linalg.norm(gp_seen - goal)), 2),
           "move_s": round(float(world.data.time) - t0, 2), "compute_s": round(time.time() - w0, 2)}
    append(path, row)
    print(f"[t04] {ctrl_name:3s} L{level:g} {i:3d} {row['status'][:30]:30s} err {row['err_true_mm']:6.1f} mm "
          f"(seen {row['err_seen_mm']:5.1f})  {row['move_s']:.1f}s", flush=True)
    arm.home()
  world.close_renderers()


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--n", type=int, default=150)
  p.add_argument("--levels", nargs="+", type=float, default=[0.0, 1.0, 2.0])
  p.add_argument("--controllers", nargs="+", default=["rl", "ik"])
  args = p.parse_args()
  path = RESULTS / "reach_sim2real.jsonl"
  tg = targets(args.n)
  for lv in args.levels:
    for c in args.controllers:
      run(c, lv, tg, path)


if __name__ == "__main__":
  main()
