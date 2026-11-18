"""Test 1 - headline comparison: VLA vs RL-based methods on randomized pick-and-place.

Every method runs the same seeded scenes in the exact simulator (level 0) and with realistic
noise (level 1: servo encoder noise + calibration bias, object mass/friction randomization,
camera noise / outliers / occlusion + extrinsic bias; see src/so101_reach/agent/sim_reality.py).

Feeds graphs: success per method (sim vs noisy), success by shape / stacked, execution time,
failure-mode breakdown, LLM calls / tokens per episode.

  python scripts/report/t01_manip_methods.py                       # defaults: 100 / 30 episodes
  python scripts/report/t01_manip_methods.py --n-fast 20 --n-llm 5 --methods rl_scripted vla
Results: results/report/manip.jsonl (shared with t02, resumable).
"""

from __future__ import annotations

import argparse

from common import DEFAULT_BASE_URL, DEFAULT_MODEL, LLM_METHODS, METHODS, RESULTS, run_grid

SEED0 = 200_000  # disjoint from BC training (0..) and grasp-eval (100000..) seeds


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
  p.add_argument("--levels", nargs="+", type=float, default=[0.0, 1.0])
  p.add_argument("--n-fast", type=int, default=100, help="episodes per (non-LLM method, level)")
  p.add_argument("--n-llm", type=int, default=30, help="episodes per (LLM method, level)")
  p.add_argument("--workers", type=int, default=8)
  p.add_argument("--model", default=DEFAULT_MODEL)
  p.add_argument("--base-url", default=DEFAULT_BASE_URL)
  args = p.parse_args()
  jobs = []
  # Interleave levels so a partial run still has both columns for every method.
  for i in range(max(args.n_fast, args.n_llm)):
    for m in args.methods:
      n = args.n_llm if m in LLM_METHODS else args.n_fast
      if i < n:
        jobs += [(m, SEED0 + i, lv) for lv in args.levels]
  run_grid(RESULTS / "manip.jsonl", jobs, workers=args.workers, model=args.model, base_url=args.base_url)


if __name__ == "__main__":
  main()
