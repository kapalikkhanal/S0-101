"""Test 7 - data efficiency of the learned grasp policy (behavior cloning on self-supervised data).

1. generate   randomized grasp episodes with scripts/gen_grasp_data.py (scripted base + perturbed
              retries, verified by a 5 cm lift; no LLM), in parallel seed shards
2. train      the BC MLP (scripts/train_bc_grasp.py) on the first N successful demonstrations,
              N in --sizes; the model trained on ALL data becomes data/il/bc_grasp.json (used by
              the rl_bc / hybrid methods of tests 1-2)
3. evaluate   first-try grasp success of each model on held-out scenes (seeds 100000+), next to
              the scripted heuristic, with and without the realistic noise level

Results: results/report/bc_efficiency.jsonl, models in results/report/bc_models/.

  python scripts/report/t07_bc_data_efficiency.py --episodes 800 --eval 100
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from common import RESULTS, ROOT, TZ, append, controller, done_keys

DATA_DIR = ROOT / "data" / "il"
SHARDS = RESULTS / "bc_shards"
MODELS = RESULTS / "bc_models"
EVAL_SEED0 = 100_000


def gen_shard(args) -> Path:
  i, n, seed = args
  out = SHARDS / f"shard{i:02d}.jsonl"
  have = len(out.read_text().splitlines()) if out.exists() else 0
  if have:
    eps = {json.loads(l)["episode"] for l in out.read_text().splitlines() if l.strip()}
    if len(eps) >= n:
      return out
    out.unlink()  # partial shard: regenerate (episodes are cheap and seeded)
  subprocess.run([sys.executable, str(ROOT / "scripts" / "gen_grasp_data.py"), "--episodes", str(n), "--seed", str(seed),
                  "--out", str(out)], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
  return out


def eval_one(args) -> dict:
  """First-try grasp + lift success of one model (or the heuristic) on one held-out scene."""
  import gen_grasp_data as G
  from types import SimpleNamespace

  from so101_reach.agent.bc_grasp import BcGrasp
  from so101_reach.agent.perception import SimPerception
  from so101_reach.agent.sim_reality import make_world, wrap_perception

  name, model_path, seed, level = args
  bc = BcGrasp.load(Path(model_path)) if model_path else None
  # Same episode as gen_grasp_data.run_episode with 0 retries, on a (possibly) noisy world.
  orig_world, orig_per = G.SimWorld, G.SimPerception
  G.SimWorld = lambda cfg: make_world(cfg, seed, level)
  G.SimPerception = lambda w: wrap_perception(SimPerception(w), seed, level, render=False)
  try:
    rows = G.run_episode(0, seed, controller(0.5), bc, SimpleNamespace(perturb_retries=0), None)
  finally:
    G.SimWorld, G.SimPerception = orig_world, orig_per
  r = rows[0]
  return {"model": name, "seed": seed, "level": level, "shape": r.get("shape"), "stacked": r.get("stacked"),
          "aborted": r.get("source") == "abort", "success": bool(r.get("success")), "note": r.get("note", "")[:120]}


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--episodes", type=int, default=800, help="generated grasp episodes in total")
  p.add_argument("--shards", type=int, default=8)
  p.add_argument("--sizes", type=int, nargs="+", default=[10, 25, 50, 100, 200, 400])
  p.add_argument("--eval", type=int, default=100, help="held-out scenes per model and level")
  p.add_argument("--levels", type=float, nargs="+", default=[0.0, 1.0])
  p.add_argument("--workers", type=int, default=8)
  args = p.parse_args()
  SHARDS.mkdir(parents=True, exist_ok=True)
  MODELS.mkdir(parents=True, exist_ok=True)
  DATA_DIR.mkdir(parents=True, exist_ok=True)

  per = args.episodes // args.shards
  print(f"[t07] generating {per * args.shards} episodes in {args.shards} shards", flush=True)
  with ProcessPoolExecutor(args.workers) as ex:
    shards = list(ex.map(gen_shard, [(i, per, i * per) for i in range(args.shards)]))
  rows = [json.loads(l) for s in shards for l in s.read_text().splitlines() if l.strip()]
  rows.sort(key=lambda r: (r["seed"], r.get("t", 0)))
  all_path = DATA_DIR / "grasp_episodes.jsonl"
  all_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
  succ = [r for r in rows if r["success"]]
  n_eps = len({r["seed"] for r in rows})
  print(f"[t07] {len(rows)} attempts in {n_eps} episodes, {len(succ)} verified grasps -> {all_path}", flush=True)
  (RESULTS / "bc_data_stats.json").write_text(json.dumps({
    "episodes": n_eps, "attempts": len(rows), "successes": len(succ),
    "by_source": {s: [sum(r["success"] for r in rows if r["source"] == s), sum(r["source"] == s for r in rows)]
                  for s in sorted({r["source"] for r in rows})}}, indent=1))

  # Train on growing prefixes of the successful demonstrations (ordered by scene seed).
  models = {"heuristic": None}
  sizes = [n for n in args.sizes if n < len(succ)] + [len(succ)]
  for n in sizes:
    sub = MODELS / f"data_{n}.jsonl"
    sub.write_text("".join(json.dumps(r) + "\n" for r in succ[:n]))
    out = MODELS / f"bc_{n}.json"
    if not out.exists():
      subprocess.run([sys.executable, str(ROOT / "scripts" / "train_bc_grasp.py"), "--data", str(sub), "--out", str(out),
                      "--min-rows", "5"], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    models[f"bc_{n}"] = out
  final = MODELS / f"bc_{len(succ)}.json"
  (DATA_DIR / "bc_grasp.json").write_text(final.read_text())
  print(f"[t07] trained {len(sizes)} BC models; full model -> {DATA_DIR / 'bc_grasp.json'}", flush=True)

  path = RESULTS / "bc_efficiency.jsonl"
  done = done_keys(path, ("model", "level", "seed"))
  jobs = [(name, str(mp) if mp else None, EVAL_SEED0 + i, lv) for lv in args.levels for name, mp in models.items()
          for i in range(args.eval) if (name, lv, EVAL_SEED0 + i) not in done]
  print(f"[t07] evaluating: {len(jobs)} grasp episodes", flush=True)
  with ProcessPoolExecutor(args.workers) as ex:
    for k, r in enumerate(ex.map(eval_one, jobs, chunksize=2)):
      r["n_demos"] = 0 if r["model"] == "heuristic" else int(r["model"].split("_")[1])
      append(path, r)
      if (k + 1) % 50 == 0:
        print(f"[t07] {k + 1}/{len(jobs)}", flush=True)
  res = [json.loads(l) for l in path.read_text().splitlines()]
  for lv in args.levels:
    for name in models:
      rs = [r for r in res if r["model"] == name and r["level"] == lv and not r["aborted"]]
      print(f"[t07] L{lv:g} {name:12s} {sum(r['success'] for r in rs)}/{len(rs)} = {np.mean([r['success'] for r in rs]):.2f}")


_ = TZ
if __name__ == "__main__":
  main()
