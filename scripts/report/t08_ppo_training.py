"""Test 8 - RL training cost and learning curves (PPO reach policy).

The released checkpoints ship without their TensorBoard logs, so this re-runs PPO training
(scripts/train.py, same task and hyper-parameters) for one or more seeds and exports the curves:
mean reward, curriculum stage / success rate / tolerance, TCP position error, policy std,
termination rates - plus wall-clock time and environment steps per iteration.

Reduced defaults fit a 6 GB laptop GPU in a few hours; pass the paper settings
(--num-envs 4096 --iterations 6000) to reproduce the released run exactly.

  python scripts/report/t08_ppo_training.py --seeds 42 --num-envs 2048 --iterations 1500
  python scripts/report/t08_ppo_training.py --export-only        # re-export curves of finished runs
Results: results/report/ppo_curves.csv, logs/rsl_rl/so101_reach/report_seed<k>/
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time

from common import RESULTS, ROOT

TAGS = {
  "reward": "Train/mean_reward",
  "episode_length": "Train/mean_episode_length",
  "stage": "Curriculum/stage/stage",
  "success": "Curriculum/stage/success_rate",
  "tol_mm": "Curriculum/stage/success_tol_mm",
  "err_m": "Metrics/ee_target/position_error",
  "std": "Policy/mean_std",
  "self_coll": "Episode_Termination/self_collision",
  "ground": "Episode_Termination/ground_contact",
  "iter_s": "Perf/total_fps",
}
LOG_ROOT = ROOT / "logs" / "rsl_rl" / "so101_reach"


def export(seeds: list[int], num_envs: int) -> None:
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

  rows = []
  for s in seeds:
    runs = sorted(p for p in LOG_ROOT.glob(f"*report_seed{s}*") if p.is_dir())
    if not runs:
      print(f"[t08] no run for seed {s}")
      continue
    ea = EventAccumulator(str(runs[-1]), size_guidance={"scalars": 0})
    ea.Reload()
    avail = set(ea.Tags()["scalars"])
    series = {}
    for k, tag in TAGS.items():
      t = tag if tag in avail else next((a for a in avail if a.endswith(tag.split("/")[-1])), None)
      if t:
        series[k] = {e.step: (e.value, e.wall_time) for e in ea.Scalars(t)}
    steps = sorted(set().union(*[set(v) for v in series.values()])) if series else []
    t0 = min(v[1] for d in series.values() for v in d.values()) if series else 0
    for it in steps:
      row = {"seed": s, "iteration": it, "env_steps": (it + 1) * num_envs * 24}
      for k, d in series.items():
        if it in d:
          row[k] = d[it][0]
          row["wall_s"] = round(d[it][1] - t0, 1)
      rows.append(row)
    print(f"[t08] seed {s}: {len(steps)} iterations, tags {sorted(series)} ({runs[-1].name})")
  if rows:
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("seed", "iteration", "env_steps", "wall_s"), k))
    out = RESULTS / "ppo_curves.csv"
    with open(out, "w", newline="") as f:
      w = csv.DictWriter(f, keys)
      w.writeheader()
      w.writerows(rows)
    print(f"[t08] -> {out}")


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--seeds", type=int, nargs="+", default=[42])
  p.add_argument("--num-envs", type=int, default=2048)
  p.add_argument("--iterations", type=int, default=1500)
  p.add_argument("--export-only", action="store_true")
  args = p.parse_args()
  if not args.export_only:
    for s in args.seeds:
      if any(LOG_ROOT.glob(f"*report_seed{s}*/model_{args.iterations - 1}.pt")):
        print(f"[t08] seed {s} already trained")
        continue
      t0 = time.time()
      subprocess.run([sys.executable, str(ROOT / "scripts" / "train.py"), "--seed", str(s), "--num-envs", str(args.num_envs),
                      "--iterations", str(args.iterations), "--run-name", f"report_seed{s}", "--no-eval"], check=True, cwd=ROOT)
      print(f"[t08] seed {s}: trained in {(time.time() - t0) / 60:.1f} min", flush=True)
  export(args.seeds, args.num_envs)


if __name__ == "__main__":
  main()
