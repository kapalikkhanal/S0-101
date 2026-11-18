"""Test 5 - decision latency / control rate of every decision-maker.

  ppo_onnx     exported PPO reach policy, ONNX Runtime CPU, batch 1 (runs every 20 ms tick)
  bc_mlp       behavior-cloned grasp MLP (numpy), one prediction per grasp
  ik_solve     the arm's damped least-squares IK pose solve (scipy), per waypoint
  vla_<model>  one vision-language agent turn: annotated head + wrist images + state -> tool call
               (local OpenAI-compatible server; measured on fresh sim scenes)

Results: results/report/latency.json (per-sample times, so the plot can show distributions).

  python scripts/report/t05_latency.py --models qwen2.5vl-tools:3b qwen2.5vl-tools:7b --n-llm 15
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from common import BOX, DEFAULT_BASE_URL, RESULTS, TZ, controller, make_llm_agent, pick_place_scene
from so101_reach.agent.arm import AgentArm
from so101_reach.agent.bc_grasp import DEFAULT_MODEL as BC_MODEL
from so101_reach.agent.bc_grasp import BcGrasp
from so101_reach.agent.perception import SimPerception
from so101_reach.agent.sim_world import SimWorld
from so101_reach.real.mouse_real import SafeTargets
from so101_reach.real.robot import SimRealPlant
from so101_reach.settings import AGENT_POLICY_DIR as POLICY
from so101_reach.sim import so101


def timeit(fn, n: int, warm: int = 20) -> list[float]:
  for _ in range(warm):
    fn()
  out = []
  for _ in range(n):
    t = time.perf_counter()
    fn()
    out.append(1000 * (time.perf_counter() - t))
  return out


def world_arm(seed: int):
  cfg, label, _ = pick_place_scene(seed)
  world = SimWorld(cfg)
  world.set_arm({**so101.HOME_JOINT_POS, **so101.FIXED_JOINT_TARGETS, "wrist_roll": world.jmap.to_rad("wrist_roll", 2048)})
  plant = SimRealPlant(world, world.jmap, world.fcal)
  plant.engage()
  for _ in range(25):
    plant.wait()
  return world, AgentArm(plant, controller(0.5), SafeTargets(), TZ), label


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--n", type=int, default=2000, help="samples for the fast decision-makers")
  p.add_argument("--models", nargs="*", default=["qwen2.5vl-tools:3b", "qwen2.5vl-tools:7b"])
  p.add_argument("--n-llm", type=int, default=15)
  p.add_argument("--base-url", default=DEFAULT_BASE_URL)
  args = p.parse_args()
  out_path = RESULTS / "latency.json"
  res = json.loads(out_path.read_text()) if out_path.exists() else {}

  import onnxruntime as ort

  sess = ort.InferenceSession(str(POLICY / "policy.onnx"), providers=["CPUExecutionProvider"])
  inp = sess.get_inputs()[0]
  x = np.zeros((1, inp.shape[1]), np.float32)
  res["ppo_onnx"] = timeit(lambda: sess.run(None, {inp.name: x}), args.n)

  world, arm, label = world_arm(300_000)
  obj = SimPerception(world).find(label)[0]
  bc = BcGrasp.load(BC_MODEL)
  if bc is not None:
    res["bc_mlp"] = timeit(lambda: bc.predict(obj, TZ), args.n)
  q6 = arm.q6()[0]
  goal = np.array([0.24, -0.05, TZ + 0.08])
  res["ik_solve"] = timeit(lambda: arm._pose(goal, None, None, q6), min(args.n, 300), warm=5)
  world.close_renderers()

  for model in args.models:
    key = f"vla_{model}"
    samples, toks = [], []
    for k in range(args.n_llm):
      world, arm, label = world_arm(300_000 + k)
      per = SimPerception(world)
      agent = make_llm_agent(arm, per, RESULTS / "agent_logs" / "latency" / f"{model.replace(':', '_')}_{k}", model,
                             args.base_url, max_turns=1)
      agent.run(f"Pick up the {label} and put it into the {BOX}, then go home.", [label, BOX], max_turns=1)
      samples += [1000 * float(t["s"] or 0) for t in agent.llm_trace]
      toks += [(t["in"] or 0, t["out"] or 0) for t in agent.llm_trace]
      world.close_renderers()
      print(f"[t05] {model} call {k + 1}/{args.n_llm}: {samples[-1]:.0f} ms, tokens {toks[-1]}", flush=True)
    res[key] = samples
    res[f"{key}_tokens"] = toks

  out_path.write_text(json.dumps(res))
  for k, v in res.items():
    if not k.endswith("_tokens"):
      print(f"[t05] {k:24s} median {np.median(v):10.3f} ms  p95 {np.percentile(v, 95):10.3f} ms  (n={len(v)})")
  print(f"[t05] -> {out_path}")


if __name__ == "__main__":
  main()
